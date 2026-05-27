# -*- coding: utf-8 -*-
"""
FMA Compaction Analyzer Pro - نسخة معدلة لاستقبال بيانات ESP32 + MPU-9255
- تستقبل بيانات الحساس مباشرة من ESP32 عبر Wi-Fi (HTTP JSON)
- تدعم GPS من المتصفح إن توفر، أو تعمل بدون GPS بنقاط محلية متسلسلة
- تحافظ على فكرة المعايرة المرجعية والتقارير والخرائط/الرسومات
"""

import io
import json
import math
import sqlite3
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:
    streamlit_js_eval = None

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FMA Compaction Analyzer Pro - ESP32",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class UnitConverter:
    UNIT_SYSTEMS = {
        "metric": {
            "name": "متري (SI)",
            "length": "متر",
            "length_short": "m",
            "density": "kg/m³",
            "area": "m²",
            "speed": "km/h",
            "to_meter": 1.0,
            "from_meter": 1.0,
            "density_factor": 1.0,
        },
        "imperial": {
            "name": "إمبراطوري",
            "length": "قدم",
            "length_short": "ft",
            "density": "pcf",
            "area": "ft²",
            "speed": "mph",
            "to_meter": 0.3048,
            "from_meter": 3.28084,
            "density_factor": 0.06242796,
        },
    }

    def __init__(self, system="metric"):
        self.system = system
        self.config = self.UNIT_SYSTEMS[system]

    def format_length(self, meters):
        value = meters * self.config["from_meter"]
        if self.system == "metric":
            return f"{value:.1f} {self.config['length_short']}" if value < 1000 else f"{value/1000:.2f} km"
        return f"{value:.1f} {self.config['length_short']}" if value < 5280 else f"{value/5280:.2f} mi"

    def to_meters(self, value):
        return value * self.config["to_meter"]


class DatabaseManager:
    def __init__(self, db_path="fma_compaction_esp32.db"):
        self.db_path = db_path
        self.init_tables()

    def init_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT UNIQUE,
                project_name TEXT,
                date TEXT,
                location TEXT,
                engineer TEXT,
                unit_system TEXT,
                data_json TEXT,
                summary_json TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def save_project(self, project_id, project_name, location, engineer, unit_system, data, summary):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            INSERT OR REPLACE INTO projects
            (project_id, project_name, date, location, engineer, unit_system, data_json, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                project_name,
                datetime.now().isoformat(),
                location,
                engineer,
                unit_system,
                json.dumps(data, ensure_ascii=False),
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()

    def get_all_projects(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT project_id, project_name, date, location FROM projects ORDER BY date DESC")
        rows = c.fetchall()
        conn.close()
        return rows

    def load_project(self, project_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT data_json, summary_json FROM projects WHERE project_id = ?", (project_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None, None
        return json.loads(row[0]), json.loads(row[1])


class CompactionCalculator:
    SOIL_FACTORS = {
        "رملية": {"energy": 1.00, "moisture_sensitivity": 0.06, "max_improvement": 1.30},
        "طينية": {"energy": 0.85, "moisture_sensitivity": 0.10, "max_improvement": 1.20},
        "غرينية": {"energy": 0.90, "moisture_sensitivity": 0.08, "max_improvement": 1.25},
        "صخرية مكسرة": {"energy": 1.15, "moisture_sensitivity": 0.04, "max_improvement": 1.15},
    }

    @staticmethod
    def get_color(value):
        if value < 40:
            return "#8B0000"
        elif value < 50:
            return "#B22222"
        elif value < 60:
            return "#DC143C"
        elif value < 65:
            return "#FF4500"
        elif value < 70:
            return "#FF6347"
        elif value < 75:
            return "#FF8C00"
        elif value < 80:
            return "#FFA500"
        elif value < 85:
            return "#FFD700"
        elif value < 88:
            return "#FFFF00"
        elif value < 91:
            return "#ADFF2F"
        elif value < 94:
            return "#7CFC00"
        elif value < 97:
            return "#32CD32"
        elif value < 100:
            return "#228B22"
        elif value < 105:
            return "#1E90FF"
        elif value < 110:
            return "#191970"
        return "#4B0082"

    @staticmethod
    def get_status(value, target_min=95, target_max=100):
        if value < target_min:
            return "🔴 غير مقبول", "poor"
        elif value <= target_max:
            return "🟢 مقبول", "good"
        return "🔵 دمك مفرط", "over"

    @staticmethod
    def estimate_from_sensor(snapshot, ref_data, ref_sensor, passes):
        soil_type = ref_data.get("soil_type", "رملية")
        soil = CompactionCalculator.SOIL_FACTORS.get(soil_type, CompactionCalculator.SOIL_FACTORS["رملية"])

        initial = float(ref_data.get("initial", 78.0))
        final = float(ref_data.get("final", 98.5))
        ref_passes = max(int(ref_data.get("passes", 8)), 1)
        omc = float(ref_data.get("omc", 12.5))
        moisture = float(ref_data.get("initial_moisture", 11.2))
        efficiency = float(ref_data.get("efficiency", 100)) / 100.0

        ref_rms = max(float(ref_sensor.get("rms", 0.05)), 0.001)
        ref_peak = max(float(ref_sensor.get("peak", 0.10)), 0.001)
        ref_freq = max(float(ref_sensor.get("dominant_hz", 15.0)), 0.10)

        cur_rms = max(float(snapshot.get("rms", 0.05)), 0.001)
        cur_peak = max(float(snapshot.get("peak", 0.10)), 0.001)
        cur_freq = max(float(snapshot.get("dominant_hz", 15.0)), 0.10)
        gyro_rms = max(float(snapshot.get("gyro_rms", 0.0)), 0.0)

        rms_ratio = cur_rms / ref_rms
        peak_ratio = cur_peak / ref_peak
        freq_ratio = cur_freq / ref_freq

        energy_current = math.log1p(max(passes, 1) * efficiency * soil["energy"])
        energy_ref = math.log1p(ref_passes * efficiency * soil["energy"])
        pass_factor = min(energy_current / max(energy_ref, 0.001), soil["max_improvement"])

        moisture_factor = math.exp(-soil["moisture_sensitivity"] * abs(moisture - omc))
        moisture_factor = max(0.65, min(1.0, moisture_factor))

        motion_penalty = 1.0 / (1.0 + 0.02 * gyro_rms)
        sensor_score = (0.50 * rms_ratio) + (0.30 * peak_ratio) + (0.20 * freq_ratio)
        sensor_score *= motion_penalty
        sensor_score = max(0.35, min(1.60, sensor_score))

        compaction = initial + ((final - initial) * sensor_score * pass_factor * moisture_factor)
        return round(max(40.0, min(compaction, 112.0)), 2)


def calculate_distance(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_browser_gps():
    if streamlit_js_eval is None:
        return None
    try:
        data = streamlit_js_eval(
            js_expressions="""
            new Promise((resolve) => {
                if (!navigator.geolocation) {
                    resolve(null);
                    return;
                }
                navigator.geolocation.getCurrentPosition(
                    (p) => resolve({
                        lat: p.coords.latitude,
                        lon: p.coords.longitude,
                        acc: p.coords.accuracy,
                        ts: Date.now()
                    }),
                    () => resolve(null),
                    {enableHighAccuracy: true, timeout: 10000, maximumAge: 0}
                );
            })
            """,
            key=f"gps_{int(time.time() * 1000)}",
        )
        return data
    except Exception:
        return None


def fetch_esp32_snapshot(base_url):
    clean_url = base_url.strip().rstrip("/")
    if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
        clean_url = f"http://{clean_url}"
    response = requests.get(f"{clean_url}/data", timeout=3)
    response.raise_for_status()
    data = response.json()
    required = ["ax", "ay", "az", "gx", "gy", "gz", "rms", "peak", "dominant_hz", "temp_c"]
    for key in required:
        if key not in data:
            raise ValueError(f"الحقل {key} غير موجود في بيانات ESP32")
    return data


def build_local_position(points, spacing_meters):
    idx = len(points) + 1
    return {
        "Point_ID": f"P{idx}",
        "Local_X_m": round((idx - 1) * spacing_meters, 2),
        "Local_Y_m": 0.0,
    }


def export_excel(df, ref_data, project_data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Compaction_Data", index=False)
        summary = pd.DataFrame(
            {
                "المؤشر": [
                    "المشروع",
                    "الموقع",
                    "المهندس",
                    "عدد النقاط",
                    "متوسط الدمك",
                    "أدنى قيمة",
                    "أعلى قيمة",
                    "مرجع RMS",
                    "مرجع Peak",
                    "مرجع Frequency",
                ],
                "القيمة": [
                    project_data.get("name", ""),
                    project_data.get("location", ""),
                    project_data.get("engineer", ""),
                    len(df),
                    f"{df['Compaction_Modulus_%'].mean():.2f}%",
                    f"{df['Compaction_Modulus_%'].min():.2f}%",
                    f"{df['Compaction_Modulus_%'].max():.2f}%",
                    ref_data.get("ref_sensor_rms", "-"),
                    ref_data.get("ref_sensor_peak", "-"),
                    ref_data.get("ref_sensor_freq", "-"),
                ],
            }
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return output.getvalue()


def export_html(df, project_data):
    good = len(df[df["Status_Type"] == "good"])
    poor = len(df[df["Status_Type"] == "poor"])
    over = len(df[df["Status_Type"] == "over"])
    return f"""
    <!DOCTYPE html>
    <html dir='rtl' lang='ar'>
    <head>
      <meta charset='UTF-8'>
      <title>تقرير الدمك الذكي</title>
      <style>
        body {{font-family: Arial, sans-serif; margin: 30px; background:#f5f5f5;}}
        .box {{max-width: 1100px; margin:auto; background:white; padding:24px; border-radius:16px;}}
        table {{width:100%; border-collapse:collapse; margin-top:20px;}}
        th, td {{border:1px solid #ddd; padding:8px; text-align:right;}}
        th {{background:#1f4e79; color:white;}}
      </style>
    </head>
    <body>
      <div class='box'>
        <h1>🏗️ تقرير الدمك الذكي</h1>
        <p><strong>المشروع:</strong> {project_data.get('name', '')}</p>
        <p><strong>الموقع:</strong> {project_data.get('location', '')}</p>
        <p><strong>التاريخ:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h2>ملخص سريع</h2>
        <ul>
          <li>عدد النقاط: {len(df)}</li>
          <li>متوسط الدمك: {df['Compaction_Modulus_%'].mean():.2f}%</li>
          <li>مقبول: {good}</li>
          <li>غير مقبول: {poor}</li>
          <li>دمك مفرط: {over}</li>
        </ul>
        {df.to_html(index=False)}
      </div>
    </body>
    </html>
    """


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.tracking_points = []
    st.session_state.is_tracking = False
    st.session_state.last_position = None
    st.session_state.passes_count = {}
    st.session_state.reference_data = None
    st.session_state.reference_sensor = None
    st.session_state.project_data = None
    st.session_state.unit_converter = UnitConverter("metric")
    st.session_state.db_manager = DatabaseManager()
    st.session_state.last_snapshot = None
    st.session_state.show_export = False


with st.sidebar:
    st.markdown("## 🏗️ FMA الدمك الذكي")
    st.markdown("### نسخة ESP32 + MPU-9255")
    st.markdown("---")

    tab_project, tab_settings, tab_saved = st.tabs(["📋 مشروع", "⚙️ إعدادات", "💾 المشاريع"])

    with tab_project:
        project_id = st.text_input("معرف المشروع", value=f"FMA-ESP32-{datetime.now().strftime('%Y%m%d%H%M')}")
        project_name = st.text_input("اسم المشروع", value="مشروع دمك ذكي تجريبي")
        project_location = st.text_input("الموقع", value="محافظة إب - اليمن")
        engineer_name = st.text_input("اسم المهندس", value="المهندس المشرف")
        layer_number = st.number_input("رقم الطبقة", min_value=1, value=1)

    with tab_settings:
        unit_system = st.selectbox(
            "نظام القياس",
            ["metric", "imperial"],
            format_func=lambda x: "🇪🇺 متري" if x == "metric" else "🇺🇸 إمبراطوري",
        )
        if unit_system != st.session_state.unit_converter.system:
            st.session_state.unit_converter = UnitConverter(unit_system)

        soil_type = st.selectbox("نوع التربة", ["رملية", "طينية", "غرينية", "صخرية مكسرة"])
        source_mode = st.selectbox("مصدر البيانات", ["ESP32 عبر Wi-Fi", "محاكاة Demo"])
        esp32_url = st.text_input("عنوان ESP32", value="192.168.4.1")
        use_browser_gps = st.checkbox("استخدام GPS من المتصفح إن توفر", value=False)
        spacing_user = st.number_input(
            f"مسافة التباعد ({st.session_state.unit_converter.config['length']})",
            min_value=0.5,
            value=5.0 if unit_system == "metric" else 16.0,
            step=0.5,
        )
        spacing_meters = st.session_state.unit_converter.to_meters(spacing_user)
        auto_interval = st.slider("فترة التحديث التلقائي (ثانية)", 1, 10, 2)
        min_accuracy = st.slider("أقل دقة GPS مقبولة (متر)", 3, 50, 15)

    with tab_saved:
        st.markdown("### المشاريع المحفوظة")
        projects = st.session_state.db_manager.get_all_projects()
        if projects:
            for proj in projects:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.caption(f"📁 **{proj[1]}**")
                    st.caption(f"🗓️ {proj[2][:10]} | 📍 {proj[3]}")
                with c2:
                    if st.button("تحميل", key=f"load_{proj[0]}"):
                        data, summary = st.session_state.db_manager.load_project(proj[0])
                        if data:
                            st.session_state.tracking_points = data.get("points", [])
                            st.success(f"✅ تم تحميل {len(st.session_state.tracking_points)} نقطة")
                            st.rerun()
                st.divider()
        else:
            st.info("لا توجد مشاريع محفوظة")


st.title("🏗️ FMA Compaction Analyzer Pro")
st.markdown("#### استقبال مباشر لبيانات **ESP32 + MPU-9255** مع حفظ وتحليل النتائج")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("📍 النقاط", len(st.session_state.tracking_points))
with m2:
    st.metric("🔌 المصدر", source_mode)
with m3:
    st.metric("🎯 التتبع", "🟢 نشط" if st.session_state.is_tracking else "⏸️ متوقف")
with m4:
    st.metric("✅ المعايرة", "مكتملة" if st.session_state.reference_data else "غير مكتملة")

st.markdown("---")

with st.expander("🔧 المعايرة المرجعية", expanded=st.session_state.reference_data is None):
    c1, c2 = st.columns(2)
    with c1:
        initial_comp = st.number_input("معامل الدمك الابتدائي (%)", 40.0, 95.0, 78.0)
        ref_passes = st.number_input("عدد الأشواط المرجعية", 1, 30, 8)
        final_comp = st.number_input("معامل الدمك النهائي المرجعي (%)", 80.0, 112.0, 98.5)
        initial_moisture = st.number_input("الرطوبة الحالية (%)", 0.0, 35.0, 11.2)
        omc = st.number_input("الرطوبة المثلى OMC (%)", 0.0, 35.0, 12.5)
        efficiency = st.slider("كفاءة المعدة (%)", 50, 120, 100)

    with c2:
        st.info("المرجع الحسي يقرأ من ESP32 والحساس في الحالة المرجعية المقبولة.")
        if st.button("📡 قراءة مرجعية من ESP32", use_container_width=True):
            try:
                if source_mode == "ESP32 عبر Wi-Fi":
                    snap = fetch_esp32_snapshot(esp32_url)
                else:
                    snap = {
                        "ax": 0.02, "ay": 0.03, "az": 1.01,
                        "gx": 0.5, "gy": 0.7, "gz": 0.2,
                        "rms": 0.12, "peak": 0.28, "dominant_hz": 16.5,
                        "gyro_rms": 0.7, "temp_c": 29.0,
                    }
                st.session_state.reference_sensor = snap
                st.success("✅ تم حفظ المرجع الحسي من ESP32")
            except Exception as e:
                st.error(f"تعذر قراءة ESP32: {e}")

        if st.session_state.reference_sensor:
            rs = st.session_state.reference_sensor
            st.json(
                {
                    "rms": rs.get("rms"),
                    "peak": rs.get("peak"),
                    "dominant_hz": rs.get("dominant_hz"),
                    "gyro_rms": rs.get("gyro_rms"),
                    "temp_c": rs.get("temp_c"),
                }
            )
        else:
            st.warning("لم تُقرأ بيانات مرجعية من ESP32 بعد")

    if st.button("✅ حفظ المعايرة", type="primary", use_container_width=True):
        st.session_state.reference_data = {
            "initial": initial_comp,
            "passes": ref_passes,
            "final": final_comp,
            "initial_moisture": initial_moisture,
            "omc": omc,
            "efficiency": efficiency,
            "soil_type": soil_type,
            "layer": layer_number,
            "ref_sensor_rms": None if not st.session_state.reference_sensor else st.session_state.reference_sensor.get("rms"),
            "ref_sensor_peak": None if not st.session_state.reference_sensor else st.session_state.reference_sensor.get("peak"),
            "ref_sensor_freq": None if not st.session_state.reference_sensor else st.session_state.reference_sensor.get("dominant_hz"),
        }
        st.session_state.project_data = {
            "id": project_id,
            "name": project_name,
            "location": project_location,
            "engineer": engineer_name,
            "layer": layer_number,
            "source_mode": source_mode,
        }
        st.success("✅ تم حفظ المعايرة بنجاح")
        st.rerun()


def capture_snapshot():
    if source_mode == "ESP32 عبر Wi-Fi":
        return fetch_esp32_snapshot(esp32_url)
    # Demo mode
    base = 0.10 + (0.03 * math.sin(time.time()))
    return {
        "ax": round(np.random.normal(0.03, 0.02), 4),
        "ay": round(np.random.normal(0.01, 0.02), 4),
        "az": round(np.random.normal(1.00, 0.03), 4),
        "gx": round(np.random.normal(0.5, 0.2), 4),
        "gy": round(np.random.normal(0.8, 0.2), 4),
        "gz": round(np.random.normal(0.4, 0.2), 4),
        "rms": round(max(0.03, np.random.normal(base, 0.02)), 4),
        "peak": round(max(0.06, np.random.normal(base * 2.3, 0.03)), 4),
        "dominant_hz": round(max(5.0, np.random.normal(16.0, 1.2)), 2),
        "gyro_rms": round(max(0.0, np.random.normal(0.9, 0.2)), 3),
        "temp_c": round(np.random.normal(30.0, 1.0), 2),
        "uptime_ms": int(time.time() * 1000),
    }


def append_measurement(snapshot):
    gps = get_browser_gps() if use_browser_gps else None
    if gps and gps.get("lat"):
        lat = float(gps["lat"])
        lon = float(gps["lon"])
        acc = float(gps.get("acc", 100.0))
        if acc > min_accuracy:
            raise ValueError(f"دقة GPS الحالية منخفضة: {acc:.1f}m")

        point_key = f"{round(lat, 5)}_{round(lon, 5)}"
        passes = st.session_state.passes_count.get(point_key, 0) + 1
        st.session_state.passes_count[point_key] = passes
        point_id = f"P{len(st.session_state.tracking_points) + 1}"
        pos = {"Point_ID": point_id, "Latitude": lat, "Longitude": lon, "Accuracy_m": round(acc, 1)}

        if st.session_state.last_position is not None:
            dist = calculate_distance(st.session_state.last_position[0], st.session_state.last_position[1], lat, lon)
            if dist < spacing_meters:
                raise ValueError(f"لم تتجاوز مسافة التباعد بعد: {dist:.2f}m")
        st.session_state.last_position = (lat, lon)
    else:
        local_pos = build_local_position(st.session_state.tracking_points, spacing_meters)
        point_key = local_pos["Point_ID"]
        passes = 1
        pos = {**local_pos, "Accuracy_m": None}

    ref_data = st.session_state.reference_data or {
        "initial": 78.0,
        "passes": 8,
        "final": 98.5,
        "initial_moisture": 11.2,
        "omc": 12.5,
        "efficiency": 100,
        "soil_type": soil_type,
    }
    ref_sensor = st.session_state.reference_sensor or {"rms": 0.12, "peak": 0.28, "dominant_hz": 16.0, "gyro_rms": 0.8}

    comp = CompactionCalculator.estimate_from_sensor(snapshot, ref_data, ref_sensor, passes)
    status_text, status_type = CompactionCalculator.get_status(comp)

    record = {
        **pos,
        "Passes": passes,
        "Compaction_Modulus_%": comp,
        "Color": CompactionCalculator.get_color(comp),
        "Status": status_text,
        "Status_Type": status_type,
        "AX_g": round(float(snapshot.get("ax", 0.0)), 4),
        "AY_g": round(float(snapshot.get("ay", 0.0)), 4),
        "AZ_g": round(float(snapshot.get("az", 0.0)), 4),
        "GX_dps": round(float(snapshot.get("gx", 0.0)), 3),
        "GY_dps": round(float(snapshot.get("gy", 0.0)), 3),
        "GZ_dps": round(float(snapshot.get("gz", 0.0)), 3),
        "RMS_g": round(float(snapshot.get("rms", 0.0)), 4),
        "Peak_g": round(float(snapshot.get("peak", 0.0)), 4),
        "Dominant_Hz": round(float(snapshot.get("dominant_hz", 0.0)), 2),
        "Gyro_RMS": round(float(snapshot.get("gyro_rms", 0.0)), 3),
        "Temp_C": round(float(snapshot.get("temp_c", 0.0)), 2),
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state.last_snapshot = snapshot
    st.session_state.tracking_points.append(record)
    return record


c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("▶️ بدء التتبع", type="primary", use_container_width=True):
        if st.session_state.reference_data is None:
            st.error("يرجى حفظ المعايرة أولاً")
        else:
            st.session_state.is_tracking = True
            st.success("تم تفعيل التتبع")
            st.rerun()

with c2:
    if st.button("📡 قراءة واحدة الآن", use_container_width=True):
        try:
            snap = capture_snapshot()
            rec = append_measurement(snap)
            st.success(f"✅ تم تسجيل {rec['Point_ID']} | الدمك = {rec['Compaction_Modulus_%']:.2f}%")
        except Exception as e:
            st.error(f"فشل القراءة: {e}")

with c3:
    if st.button("⏹️ إيقاف", use_container_width=True):
        st.session_state.is_tracking = False
        st.warning("تم إيقاف التتبع")

with c4:
    if st.button("💾 حفظ المشروع", use_container_width=True):
        if st.session_state.tracking_points:
            data = {"points": st.session_state.tracking_points}
            summary = {"total_points": len(st.session_state.tracking_points)}
            st.session_state.db_manager.save_project(
                project_id,
                project_name,
                project_location,
                engineer_name,
                unit_system,
                data,
                summary,
            )
            st.success("✅ تم حفظ المشروع")
        else:
            st.warning("لا توجد بيانات لحفظها")

with c5:
    if st.button("🗑️ مسح", use_container_width=True):
        st.session_state.tracking_points = []
        st.session_state.last_position = None
        st.session_state.passes_count = {}
        st.session_state.last_snapshot = None
        st.success("تم مسح البيانات")
        st.rerun()


if st.session_state.is_tracking:
    st.info(f"📡 التتبع النشط: قراءة من {source_mode} كل {auto_interval} ثانية")
    try:
        snap = capture_snapshot()
        rec = append_measurement(snap)
        st.success(f"تم تسجيل {rec['Point_ID']} تلقائياً")
    except Exception as e:
        st.warning(f"تعذر تسجيل القراءة الحالية: {e}")
    time.sleep(auto_interval)
    st.rerun()


if st.session_state.last_snapshot:
    st.markdown("### 📟 آخر قراءة من ESP32")
    snap = st.session_state.last_snapshot
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMS", f"{snap.get('rms', 0):.4f} g")
    c2.metric("Peak", f"{snap.get('peak', 0):.4f} g")
    c3.metric("Dominant Hz", f"{snap.get('dominant_hz', 0):.2f}")
    c4.metric("Gyro RMS", f"{snap.get('gyro_rms', 0):.3f}")
    c5.metric("Temp", f"{snap.get('temp_c', 0):.1f} °C")


if st.session_state.tracking_points:
    df = pd.DataFrame(st.session_state.tracking_points)
    st.subheader(f"📍 البيانات المسجلة ({len(df)} نقطة)")

    preferred_cols = [
        "Point_ID", "Latitude", "Longitude", "Local_X_m", "Passes",
        "Compaction_Modulus_%", "RMS_g", "Peak_g", "Dominant_Hz", "Status", "Timestamp"
    ]
    visible_cols = [c for c in preferred_cols if c in df.columns]
    st.dataframe(df[visible_cols], use_container_width=True, height=260)

    has_gps = "Latitude" in df.columns and df["Latitude"].notna().any()
    st.markdown("### 🗺️ العرض المكاني")
    if has_gps:
        gps_df = df[df["Latitude"].notna()].copy()
        fig = px.scatter_mapbox(
            gps_df,
            lat="Latitude",
            lon="Longitude",
            color="Compaction_Modulus_%",
            size=[15] * len(gps_df),
            size_max=22,
            zoom=17,
            center={"lat": gps_df["Latitude"].mean(), "lon": gps_df["Longitude"].mean()},
            mapbox_style="carto-positron",
            color_continuous_scale="Turbo",
            hover_data={"Point_ID": True, "Passes": True, "Compaction_Modulus_%": ':.2f'},
            title="خريطة نقاط الدمك",
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=gps_df["Latitude"].tolist(),
                lon=gps_df["Longitude"].tolist(),
                mode="lines+markers",
                line=dict(width=2, color="gray"),
                marker=dict(size=8, color="black"),
                name="المسار",
            )
        )
        fig.update_layout(height=550, margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        if "Local_X_m" not in df.columns:
            df["Local_X_m"] = np.arange(len(df)) * spacing_meters
        fig_local = px.scatter(
            df,
            x="Local_X_m",
            y="Compaction_Modulus_%",
            color="Status_Type",
            color_discrete_map={"good": "green", "poor": "red", "over": "blue"},
            hover_data=[c for c in ["Point_ID", "RMS_g", "Peak_g", "Dominant_Hz"] if c in df.columns],
            title="الدمك مقابل الموقع المحلي",
            labels={"Local_X_m": "المسافة المحلية (m)", "Compaction_Modulus_%": "معامل الدمك (%)"},
        )
        fig_local.update_traces(marker=dict(size=12))
        st.plotly_chart(fig_local, use_container_width=True)

    st.markdown("### 📊 التحليلات")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("المتوسط", f"{df['Compaction_Modulus_%'].mean():.2f}%")
    s2.metric("الأدنى", f"{df['Compaction_Modulus_%'].min():.2f}%")
    s3.metric("الأعلى", f"{df['Compaction_Modulus_%'].max():.2f}%")
    s4.metric("الانحراف", f"{df['Compaction_Modulus_%'].std():.2f}" if len(df) > 1 else "0.00")
    s5.metric("المقبول", f"{len(df[df['Status_Type']=='good'])}/{len(df)}")

    h1, h2 = st.columns(2)
    with h1:
        fig_hist = px.histogram(
            df,
            x="Compaction_Modulus_%",
            nbins=20,
            title="توزيع قيم الدمك",
            labels={"Compaction_Modulus_%": "معامل الدمك (%)", "count": "عدد النقاط"},
        )
        fig_hist.add_vline(x=95, line_dash="dash", line_color="green", annotation_text="حد القبول 95%")
        st.plotly_chart(fig_hist, use_container_width=True)

    with h2:
        fig_rms = px.line(
            df,
            x="Point_ID",
            y=[c for c in ["RMS_g", "Peak_g", "Dominant_Hz"] if c in df.columns],
            markers=True,
            title="سلوك الإشارة الحسية عبر النقاط",
        )
        st.plotly_chart(fig_rms, use_container_width=True)

    st.markdown("### 📄 التصدير")
    x1, x2 = st.columns(2)
    with x1:
        excel_data = export_excel(df, st.session_state.reference_data or {}, st.session_state.project_data or {})
        st.download_button(
            "📊 تنزيل Excel",
            data=excel_data,
            file_name=f"report_{project_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with x2:
        html_data = export_html(df, st.session_state.project_data or {})
        st.download_button(
            "📄 تنزيل HTML",
            data=html_data,
            file_name=f"report_{project_id}.html",
            mime="text/html",
        )


with st.expander("📖 ملاحظات التشغيل", expanded=False):
    st.markdown(
        """
        1. شغّل كود ESP32 واربط الحساس MPU-9255 على I2C.
        2. افتح تطبيق Streamlit ثم أدخل عنوان ESP32.
        3. اقرأ المرجع الحسي في منطقة معروفة الدمك، ثم احفظ المعايرة.
        4. استخدم "قراءة واحدة" أو فعّل التتبع التلقائي.
        5. إن لم يتوفر GPS فسيعمل البرنامج على نقاط محلية متسلسلة.
        6. القيم المحسوبة هنا عملية/تجريبية وليست بديلاً عن المعايرة الحقلية والاختبارات المخبرية.
        """
    )

st.markdown("---")
st.caption(f"FMA Compaction Analyzer Pro - ESP32 Edition | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
