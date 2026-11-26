import os
from fastapi import HTTPException 
import glob
import time
import json
from typing import Optional
import cv2
from fastapi import FastAPI, File, Form, UploadFile
from google import genai
from google.genai import types
import numpy as np
from pypdf import PdfReader
from PIL import Image
import io
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass
app = FastAPI(title="Master AI Hastrekha & Samudrik Shastra Scanner")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Paste your Gemini API Key inside the quotes below
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)


def load_all_pdfs_from_folder():
    pdf_files = sorted(glob.glob("*.pdf"))
    if not pdf_files:
        return "No local PDF files found."

    combined_text = ""
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            combined_text += f"\n--- SOURCE: {pdf_file} ---\n"
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    combined_text += text + "\n"
        except Exception as e:
            combined_text += f"\n[Error reading {pdf_file}: {str(e)}]\n"

    return combined_text


ALL_PDF_KNOWLEDGE = load_all_pdfs_from_folder()


def prepare_image_for_gemini(contents: bytes, filename: str):
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
            try:
                pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return None, None
    if img is not None:
        orig_h, orig_w = img.shape[:2]
        target_w = 1280
        scale = target_w / orig_w
        resized = cv2.resize(img, (target_w, int(orig_h * scale)))

        # High-precision CLAHE enhancement to reveal micro-lines, forks, and islands
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        _, buffer = cv2.imencode(
            ".jpg", enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        return buffer.tobytes(), "image/jpeg"

    ext = filename.lower()
    if ext.endswith(".heic") or ext.endswith(".heif"):
        return contents, "image/heic"

    return None, None


@app.post("/bhavishyafal")
async def get_master_bhavishyafal(
    question_topic: str = Form(
        default="Vivah, Career, Dhan aur Sampoorna Jeevan ka sanyukt vishleshan",
        description="Apna prashn likhen (e.g. Vivah, Career, Dhan, Health)",
    ),
    gender: str = Form(
        default="Not Specified",
        description="Optional: Male (Purush) ya Female (Mahila)",
    ),
    right_hand: UploadFile = File(
        ...,
        description=(
            "Right Hand Photo (दायाँ हाथ - Karm aur Vartaman Jeevan) [Required]"
        ),
    ),
    left_hand: Optional[UploadFile] = File(
        None,
        description=(
            "Left Hand Photo (बायाँ हाथ - Janmjat Bhagya) [Optional - Poora"
            " tulnatmak vishleshan ke liye]"
        ),
    ),
):
    content_parts = []
    hands_uploaded_count = 0

    # Process Right Hand (Primary Required Image)
    c_right = await right_hand.read()
    b_right, m_right = prepare_image_for_gemini(
        c_right, right_hand.filename or ""
    )
    if b_right is None:
        return {
            "error": (
                "Right Hand image format unsupported. Please upload JPG, PNG,"
                " or HEIC."
            )
        }
    content_parts.append("UPLOADED AS RIGHT HAND (Primary Active Hand):")
    content_parts.append(types.Part.from_bytes(data=b_right, mime_type=m_right))
    hands_uploaded_count += 1

    # Process Left Hand (Optional Secondary Image)
    if left_hand and left_hand.filename:
        c_left = await left_hand.read()
        if len(c_left) > 0:
            b_left, m_left = prepare_image_for_gemini(
                c_left, left_hand.filename or ""
            )
            if b_left is not None:
                content_parts.append(
                    "UPLOADED AS LEFT HAND (Secondary Inherited Hand):"
                )
                content_parts.append(
                    types.Part.from_bytes(data=b_left, mime_type=m_left)
                )
                hands_uploaded_count += 1

    prompt = f"""
    You are a Supreme Grandmaster of Indian Samudrik Shastra (Hastrekha Vigyan).
    Perform an uncompromising, razor-sharp, deeply perceptive palmistry analysis of the attached image(s).

    USER DETAILS & QUESTION:
    - Focus Question: {question_topic}
    - Gender: {gender}
    - Total Hands Uploaded: {hands_uploaded_count}

    REFERENCE RULES FROM USER'S 3 PDF BOOKS (Bhag 1: Parvat/Rekha, Bhag 2: Age/Timing, Bhag 3: Upay/Karma):
    {ALL_PDF_KNOWLEDGE[:25000]}

    STRICT ANALYTICAL & ANATOMICAL RULES (FOLLOW 100%):
    1. **ANATOMICAL VERIFICATION (Thumb Orientation Rule):**
       - When a palm faces the camera: Thumb on the **LEFT side** = **RIGHT HAND (Daaya Haath)**. Thumb on the **RIGHT side** = **LEFT HAND (Baaya Haath)**.
       - Verify whether the user uploaded the correct hands in the slots. If they accidentally swapped them, silently use the true anatomical hand (Left vs Right) for your reading and politely note it in the comparison section.
       - If ONLY 1 hand is uploaded: Give a complete, deep reading from that hand.
       - If BOTH hands are uploaded: Compare Left Hand (inherited destiny, subconscious patterns) with Right Hand (current karma, struggles, and real achievements).
    2. **ZERO GENERIC HOROSCOPE FLUFF (Strictly Forbidden):**
       - NEVER write vague lines like "you are hardworking, practical, and will succeed."
       - Look at the ACTUAL physical features of this palm: dense intersecting Rahu/stress lines, slope of the Head line toward Moon mount, grid lines on Venus mount, breaks or islands, gaps between fingers, thumb flexibility, and mount prominence.
    3. **SECTION 1 - EXACT PERSONALITY & INNER MIND MIRROR (Trust Builder):**
       - Reveal their true psychology: how they think under pressure, hidden emotional wounds, trust patterns, overthinking habits, anger triggers, and genuine strengths—so specific that the user feels you are reading their mind.
    4. **SECTION 2 - BIGGEST CORE PROBLEM & RECURRING OBSTACLES (Raw Truth):**
       - Clearly identify the #1 recurring problem in their life, mind, career, or relationships visible on the palm (e.g., scattered energy from Rahu cross-lines, emotional sensitivity, career delays, or relationship friction) and explain the exact palmistry cause.
    5. **SECTION 3 - GOOD vs BAD YOGAS & COMBINED PREDICTION:**
       - Trace every major line (Life, Head, Heart, Fate, Sun, Marriage) from origin to destination mount, noting thickness, forks (do-mukhi), and breaks.
       - Clearly distinguish STRONG/LUCKY yogas from WEAK/BLOCKED areas, and answer '{question_topic}' by combining all relevant mounts and lines.
    6. **SECTION 4 - AGE MILESTONES & TARGETED REMEDIES:**
       - Calculate concrete age brackets (e.g., 24-27, 30-33, 36-40) using geometric line intersections and provide targeted remedies from Bhag 3 PDF.

    Return strictly valid JSON in both Hindi (Devanagari) and English:
    {{
        "swabhav_vyavahar_aur_antarmann": {{
            "hindi": "Sharp, non-generic mirror of their exact psychology, hidden emotional struggles, habits, and strengths in Hindi",
            "english": "Sharp, non-generic mirror of their exact psychology and habits in English"
        }},
        "haath_ki_pehchan_aur_tulna": {{
            "hindi": "Verification of Right Hand (karma) and Left Hand (destiny) and deep comparison of what changed between them in Hindi",
            "english": "Verification of Right and Left hands and deep comparison in English"
        }},
        "sabse_badi_samasya_aur_bar_bar_hone_wali_rukawat": {{
            "hindi": "Direct, unfiltered explanation of their #1 biggest life/mental/career problem and recurring obstacles visible in the lines, plus why it happens in Hindi",
            "english": "Direct explanation of their #1 biggest problem and recurring obstacles in English"
        }},
        "rekhaon_ka_sukshma_nirikshan_achha_aur_bura": {{
            "hindi": "Detailed path of lines (origin, destination, breaks, forks, Rahu web, Mounts) clearly separating GOOD yogas from BAD doshas in Hindi",
            "english": "Detailed path of lines clearly separating GOOD yogas from BAD doshas in English"
        }},
        "sanyukt_bhavishyafal_aur_umra_ganana": {{
            "hindi": "Combined prediction for {question_topic} with specific age milestones and calculation logic in Hindi",
            "english": "Combined prediction with specific age milestones in English"
        }},
        "upay_aur_karm_sudhar": {{
            "hindi": "Targeted remedies for the exact doshas found in Hindi",
            "english": "Targeted remedies in English"
        }}
    }}
    """

    content_parts.append(prompt)

    prediction_result = None
    last_error = ""
   
    # Prioritize deep reasoning models first (Pro / full Flash) over Lite models
    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
    ]
    try:
        for m in client.models.list():
            if "pro" in m.name.lower() or "flash" in m.name.lower():
                if m.name not in preferred_models:
                    preferred_models.append(m.name)
    except Exception as e:
        last_error = f"List models error: {str(e)}"
    
        prediction_result = None
    for round_num in range(2):
        for model_name in preferred_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=content_parts,
                )
                clean_json = (
                    response.text.replace("```json", "").replace("```", "").strip()
                )
                prediction_result = json.loads(clean_json)
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(3)
        if prediction_result:
            break

    if not prediction_result:
        prediction_result = {
            "error": "Gemini Vision prediction failed.",
            "models_tried": preferred_models,
            "details": last_error,
        }

    return {
        "status": "success",
        "hands_analyzed": hands_uploaded_count,
        "pdfs_loaded": len(glob.glob("*.pdf")),
        "hastrekha_report": prediction_result,
    }
from fastapi.responses import FileResponse

@app.get("/")
def serve_website():
    return FileResponse("index.html")
import sqlite3
import razorpay
from datetime import datetime, timedelta
from pydantic import BaseModel

# --- Razorpay Keys (इंटरनेट पर डालने से पहले Environment Variables में रखें) ---
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")

rzp_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# --- SQLite Database for 3-Day Free Trial & VIP Memberships ---
def init_db():
    conn = sqlite3.connect("users.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memberships (
            phone TEXT PRIMARY KEY,
            plan_name TEXT,
            expires_at TEXT,
            single_credits INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()

PLANS = {
    "single_49": {"amount": 4900, "days": 0, "credits": 1, "name": "1-Time Full Report (₹49)"},
    "vip_1m": {"amount": 24900, "days": 30, "credits": 0, "name": "1 Month VIP Unlimited (₹249)"},
    "vip_2m": {"amount": 39900, "days": 60, "credits": 0, "name": "2 Months VIP Unlimited (₹399)"},
    "vip_6m": {"amount": 49900, "days": 180, "credits": 0, "name": "6 Months Maha-VIP (₹499)"},
}

class PhoneReq(BaseModel):
    phone: str

class OrderReq(BaseModel):
    phone: str
    plan_id: str

class VerifyReq(BaseModel):
    phone: str
    plan_id: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

@app.post("/api/check-access")
def check_or_start_trial(req: PhoneReq):
    phone = req.phone.strip()
    if len(phone) < 10:
        raise HTTPException(status_code=400, detail="सही 10 अंकों का मोबाइल नंबर डालें")

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT plan_name, expires_at, single_credits FROM memberships WHERE phone = ?", (phone,))
    row = cur.fetchone()
    now = datetime.now()

    if not row:
        # नए यूजर के लिए 3 दिन का फ्री ट्रायल तुरंत चालू करें
        expiry = now + timedelta(days=3)
        cur.execute(
            "INSERT INTO memberships (phone, plan_name, expires_at, single_credits) VALUES (?, ?, ?, ?)",
            (phone, "3-Day Free Trial", expiry.isoformat(), 0)
        )
        conn.commit()
        conn.close()
        return {"unlocked": True, "plan": "3-Day Free Trial (नया यूजर ऑफर)", "expires_at": expiry.strftime("%d %b %Y, %I:%M %p")}

    plan_name, expires_at, credits = row
    exp_dt = datetime.fromisoformat(expires_at)

    if exp_dt > now:
        conn.close()
        return {"unlocked": True, "plan": plan_name, "expires_at": exp_dt.strftime("%d %b %Y, %I:%M %p")}

    if credits > 0:
        cur.execute("UPDATE memberships SET single_credits = single_credits - 1 WHERE phone = ?", (phone,))
        conn.commit()
        conn.close()
        return {"unlocked": True, "plan": "Single Report Credit", "expires_at": "1-Time Unlocked"}

    conn.close()
    return {"unlocked": False, "plan": "Expired", "expires_at": exp_dt.strftime("%d %b %Y")}

@app.post("/api/create-order")
def create_razorpay_order(req: OrderReq):
    if req.plan_id not in PLANS:
        raise HTTPException(status_code=400, detail="Invalid Plan")
    plan = PLANS[req.plan_id]
    order = rzp_client.order.create({
        "amount": plan["amount"],
        "currency": "INR",
        "payment_capture": 1,
        "notes": {"phone": req.phone, "plan_id": req.plan_id}
    })
    return {"order_id": order["id"], "amount": plan["amount"], "key_id": RAZORPAY_KEY_ID, "plan_name": plan["name"]}

@app.post("/api/verify-payment")
def verify_razorpay_payment(req: VerifyReq):
    try:
        rzp_client.utility.verify_payment_signature({
            "razorpay_order_id": req.razorpay_order_id,
            "razorpay_payment_id": req.razorpay_payment_id,
            "razorpay_signature": req.razorpay_signature
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Payment verification failed!")

    plan = PLANS[req.plan_id]
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    now = datetime.now()

    if plan["days"] > 0:
        expiry = now + timedelta(days=plan["days"])
        cur.execute(
            "INSERT OR REPLACE INTO memberships (phone, plan_name, expires_at, single_credits) VALUES (?, ?, ?, 0)",
            (req.phone.strip(), plan["name"], expiry.isoformat())
        )
    else:
        cur.execute("SELECT expires_at FROM memberships WHERE phone = ?", (req.phone.strip(),))
        row = cur.fetchone()
        old_exp = row[0] if row else now.isoformat()
        cur.execute(
            "INSERT OR REPLACE INTO memberships (phone, plan_name, expires_at, single_credits) VALUES (?, ?, ?, 0)",
            (req.phone.strip(), plan["name"], old_exp)
        )

    conn.commit()
    conn.close()
    return {"status": "unlocked", "plan": plan["name"]}
