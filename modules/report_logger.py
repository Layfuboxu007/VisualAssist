import os
import time
from datetime import datetime

class ReportLogger:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.events = []
        
        # Ensure reports directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def log_event(self, event_type, data):
        """Logs an event during system runtime."""
        current_time = time.time()
        time_str = datetime.now().strftime("%I:%M:%S %p")
        event = {
            "timestamp": current_time,
            "time_str": time_str,
            "type": event_type.lower(),
            "data": data
        }
        self.events.append(event)
        print(f"[LOGGER] Logged {event_type.upper()}: {data}")

    def generate_report(self):
        """Generates a Markdown and a premium HTML report summarizing the session."""
        end_time = time.time()
        end_datetime = datetime.now()
        duration_sec = int(end_time - self.start_time)
        
        # Calculate summary metrics
        total_obstacles = 0
        critical_alerts = 0
        faces_seen = {}
        ocr_texts = []
        colors_identified = []
        emergencies = []
        
        # Track obstacles by label to avoid double counting same-frame detections
        # For simplicity, we just count events
        for ev in self.events:
            ev_type = ev["type"]
            data = ev["data"]
            
            if ev_type == "obstacle_warning":
                total_obstacles += 1
                if data.get("is_critical", False):
                    critical_alerts += 1
            elif ev_type == "face_recognition":
                name = data.get("name", "Unknown")
                if name != "Unknown":
                    faces_seen[name] = faces_seen.get(name, 0) + 1
            elif ev_type == "ocr":
                ocr_texts.append(data.get("text", ""))
            elif ev_type == "color_identification":
                colors_identified.append(data.get("color", ""))
            elif ev_type == "emergency":
                emergencies.append({
                    "time": ev["time_str"],
                    "reason": data.get("reason", "Unknown"),
                    "gps": data.get("gps", "N/A")
                })
                
        timestamp_str = end_datetime.strftime("%Y%m%d_%H%M%S")
        
        # 1. Generate Markdown Report
        md_filename = f"guardian_report_{timestamp_str}.md"
        md_path = os.path.join(self.output_dir, md_filename)
        md_content = self._build_markdown(
            duration_sec, total_obstacles, critical_alerts, 
            faces_seen, ocr_texts, colors_identified, emergencies
        )
        
        # 2. Generate HTML Report
        html_filename = f"guardian_report_{timestamp_str}.html"
        html_path = os.path.join(self.output_dir, html_filename)
        html_content = self._build_html(
            duration_sec, total_obstacles, critical_alerts, 
            faces_seen, ocr_texts, colors_identified, emergencies
        )
        
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"[LOGGER] Generated guardian reports:\n - Markdown: {md_path}\n - HTML: {html_path}")
            return md_path, html_path
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to write reports: {e}")
            return None, None

    def _build_markdown(self, duration_sec, total_obstacles, critical_alerts, 
                        faces_seen, ocr_texts, colors_identified, emergencies):
        duration_str = f"{duration_sec // 60}m {duration_sec % 60}s" if duration_sec >= 60 else f"{duration_sec}s"
        
        faces_str = "\n".join([f"- **{name}**: seen {count} time(s)" for name, count in faces_seen.items()])
        if not faces_str:
            faces_str = "None registered seen."
            
        ocr_str = "\n".join([f"- \"{text}\"" for text in ocr_texts])
        if not ocr_str:
            ocr_str = "No text read."
            
        colors_str = "\n".join([f"- {color}" for color in colors_identified])
        if not colors_str:
            colors_str = "No color inquiries."
            
        emergencies_str = "\n".join([f"- **{em['time']}**: {em['reason']} (GPS: {em['gps']})" for em in emergencies])
        if not emergencies_str:
            emergencies_str = "No emergency conditions encountered."
            
        # Standard guardian tips based on logs
        recs = []
        if critical_alerts > 5:
            recs.append("- User encountered a high frequency of close-proximity obstacles. Recommend clearing indoor path clutter or selecting wider outdoor routes.")
        if emergencies:
            recs.append("- Emergency alarms were triggered. Please inspect the emergency evidence images and verify user's mobility health.")
        if not recs:
            recs.append("- Navigation session was completed safely. No abnormal threat patterns detected.")

        recs_str = "\n".join(recs)
        
        return f"""# SightAssist Guardian Report
Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}

## 1. Session Overview
- **Start Time**: {self.start_datetime.strftime('%Y-%m-%d %I:%M:%S %p')}
- **Duration**: {duration_str}
- **Security Status**: {"⚠️ DANGER (Emergency triggered)" if emergencies else "✅ SECURE"}

## 2. Navigation & Obstacles
- **Total Obstacles Detected**: {total_obstacles}
- **Critical Proximity Warnings**: {critical_alerts}

## 3. Social Interaction (Faces Recognized)
{faces_str}

## 4. Utilities Used
### Text Read (OCR)
{ocr_str}

### Colors Identified
{colors_str}

## 5. Emergency Incident Log
{emergencies_str}

## 6. Guardian Recommendations
{recs_str}
"""

    def _build_html(self, duration_sec, total_obstacles, critical_alerts, 
                    faces_seen, ocr_texts, colors_identified, emergencies):
        duration_str = f"{duration_sec // 60}m {duration_sec % 60}s" if duration_sec >= 60 else f"{duration_sec}s"
        status_class = "status-danger" if emergencies else "status-secure"
        status_text = "DANGER / ALERTS" if emergencies else "SECURE & STABLE"
        
        # Build tables / lists
        faces_rows = "".join([f"<tr><td><strong>{name}</strong></td><td>{count} time(s)</td></tr>" for name, count in faces_seen.items()])
        if not faces_rows:
            faces_rows = "<tr><td colspan='2' class='text-center text-muted'>No registered faces seen during session</td></tr>"
            
        ocr_rows = "".join([f"<tr><td>{datetime.now().strftime('%I:%M %p')}</td><td>\"{text}\"</td></tr>" for text in ocr_texts])
        if not ocr_rows:
            ocr_rows = "<tr><td colspan='2' class='text-center text-muted'>No text read</td></tr>"
            
        colors_rows = "".join([f"<li>{color}</li>" for color in colors_identified])
        if not colors_rows:
            colors_rows = "<li class='text-muted'>No colors identified</li>"
            
        em_rows = "".join([f"<tr class='table-danger-row'><td>{em['time']}</td><td><strong>{em['reason']}</strong></td><td>{em['gps']}</td></tr>" for em in emergencies])
        if not em_rows:
            em_rows = "<tr><td colspan='3' class='text-center text-muted'>No emergencies triggered</td></tr>"

        recs = []
        if critical_alerts > 5:
            recs.append("User encountered a high frequency of close-proximity obstacles. Recommend clearing indoor path clutter or selecting wider outdoor routes.")
        if emergencies:
            recs.append("Emergency alarms were triggered. Please inspect the emergency evidence images and verify user's mobility health.")
        if not recs:
            recs.append("Navigation session was completed safely. No abnormal threat patterns detected.")
        recs_list = "".join([f"<li>{rec}</li>" for rec in recs])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SightAssist Guardian Session Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.03);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #5271ff;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --text-main: #f3f4f6;
            --text-secondary: #9ca3af;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}
        
        .container {{
            max-width: 900px;
            width: 100%;
            background: linear-gradient(145deg, rgba(20, 27, 45, 0.9), rgba(10, 15, 25, 0.95));
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
        }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 25px;
            margin-bottom: 30px;
        }}
        
        h1 {{
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #ffffff, var(--primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .report-meta {{
            font-size: 0.95rem;
            color: var(--text-secondary);
            text-align: right;
        }}
        
        .status-badge {{
            padding: 8px 16px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.05rem;
            display: inline-block;
        }}
        
        .status-secure {{
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}
        
        .status-danger {{
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 0.7; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.7; }}
        }}

        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(82, 113, 255, 0.3);
        }}
        
        .stat-num {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin: 5px 0;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05rem;
        }}
        
        .section-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 40px;
            margin-bottom: 15px;
            color: #ffffff;
            border-left: 4px solid var(--primary);
            padding-left: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 20px;
        }}
        
        th {{
            background-color: rgba(255,255,255,0.02);
            color: var(--text-secondary);
            font-weight: 600;
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-main);
        }}
        
        tr:hover {{
            background-color: rgba(255,255,255,0.01);
        }}
        
        .table-danger-row {{
            background-color: rgba(239, 68, 68, 0.05);
        }}
        
        .table-danger-row:hover {{
            background-color: rgba(239, 68, 68, 0.08);
        }}
        
        ul {{
            margin: 10px 0;
            padding-left: 20px;
            color: var(--text-main);
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        .rec-box {{
            background-color: rgba(82, 113, 255, 0.05);
            border: 1px solid rgba(82, 113, 255, 0.15);
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
        }}
        
        .rec-box li {{
            color: #e5e7eb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>SightAssist Guardian Report</h1>
                <div class="status-badge {status_class}" style="margin-top: 8px;">
                    {status_text}
                </div>
            </div>
            <div class="report-meta">
                <strong>Session Date:</strong> {self.start_datetime.strftime('%B %d, %Y')}<br>
                <strong>Generated At:</strong> {datetime.now().strftime('%I:%M %p')}<br>
                <strong>Start Time:</strong> {self.start_datetime.strftime('%I:%M:%S %p')}
            </div>
        </header>
        
        <div class="grid-3">
            <div class="stat-card">
                <div class="stat-label">Session Duration</div>
                <div class="stat-num" style="color: #ffffff;">{duration_str}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Obstacles Encountered</div>
                <div class="stat-num">{total_obstacles}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Critical Proximity Warnings</div>
                <div class="stat-num" style="color: {"var(--danger)" if critical_alerts > 0 else "var(--primary)"};">{critical_alerts}</div>
            </div>
        </div>
        
        <div class="section-title">Social Interaction (Faces Recognized)</div>
        <table>
            <thead>
                <tr>
                    <th>Person Name</th>
                    <th>Frequency Seen</th>
                </tr>
            </thead>
            <tbody>
                {faces_rows}
            </tbody>
        </table>
        
        <div class="section-title">Utilities & Context Queries</div>
        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px;">
            <div>
                <strong style="color: var(--text-secondary);">Text Read (OCR) History:</strong>
                <table style="font-size: 0.95rem;">
                    <thead>
                        <tr>
                            <th style="width: 25%;">Time</th>
                            <th>Extracted Content</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ocr_rows}
                    </tbody>
                </table>
            </div>
            <div>
                <strong style="color: var(--text-secondary);">Colors Identified:</strong>
                <ul style="margin-top: 15px;">
                    {colors_rows}
                </ul>
            </div>
        </div>
        
        <div class="section-title">Emergency Incident Log</div>
        <table>
            <thead>
                <tr>
                    <th>Alert Time</th>
                    <th>Trigger Reason</th>
                    <th>GPS Coordinates</th>
                </tr>
            </thead>
            <tbody>
                {em_rows}
            </tbody>
        </table>
        
        <div class="rec-box">
            <h3 style="margin-top: 0; color: #ffffff; font-weight: 600;">Guardian Safety Insights</h3>
            <ul>
                {recs_list}
            </ul>
        </div>
    </div>
</body>
</html>
"""
        return html
