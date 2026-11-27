import os
import glob
import time
import json
import io
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np
import razorpay
from PIL import Image
from pypdf import PdfReader
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

app = FastAPI(title="Master AI Hastrekha & Samudrik Shastra Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")
rzp_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


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
            import pillow_heif
            heif_file = pillow_heif.read_heif(contents)
            pil_img = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            ).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            try:
                pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

    if img is not None:
        orig_h, orig_w = img.shape[:2]
        target_w = 1280
        scale = target_w / orig_w
        resized = cv2.resize(img, (target_w, int(orig_h * scale)))

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
    if ext.endswith(".webp"):
        return contents, "image/webp"
    if ext.endswith(".png"):
        return contents, "image/png"
    return contents, "image/jpeg"
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
        description="Right Hand Photo (दायाँ हाथ - Karm aur Vartaman Jeevan) [Required]",
    ),
    left_hand: Optional[UploadFile] = File(
        None,
        description="Left Hand Photo (बायाँ हाथ - Janmjat Bhagya) [Optional]",
    ),
):
    content_parts = []
    hands_uploaded_count = 0

    c_right = await right_hand.read()
    b_right, m_right = prepare_image_for_gemini(c_right, right_hand.filename or "")
    if b_right is None:
        return {"error": "Right Hand image format unsupported. Please upload JPG, PNG, or HEIC."}
    content_parts.append("UPLOADED AS RIGHT HAND (Primary Active Hand):")
    content_parts.append(types.Part.from_bytes(data=b_right, mime_type=m_right))
    hands_uploaded_count += 1

    if left_hand and left_hand.filename:
        c_left = await left_hand.read()
        if len(c_left) > 0:
            b_left, m_left = prepare_image_for_gemini(c_left, left_hand.filename or "")
            if b_left is not None:
                content_parts.append("UPLOADED AS LEFT HAND (Secondary Inherited Hand):")
                content_parts.append(types.Part.from_bytes(data=b_left, mime_type=m_left))
                hands_uploaded_count += 1

    prompt = f"""
    You are a Supreme Grandmaster of Indian Samudrik Shastra (Hastrekha Vigyan).
    Analyze ONLY the specific physical palm image(s) attached. Every human hand is unique—your output MUST reflect the exact, unique visual reality of THIS photo, never a standard template.

    USER DETAILS & QUESTION:
    - Focus Question: {question_topic}
    - Gender: {gender}
    - Total Hands Uploaded: {hands_uploaded_count}

    REFERENCE RULES FROM USER'S 3 PDF BOOKS (Use strictly as a rulebook to interpret what you visually see, DO NOT copy generic text):
    {ALL_PDF_KNOWLEDGE[:18000]}

    STRICT VISUAL-FIRST PALMISTRY RULES:
    1. **OBSERVE PHYSICAL ANATOMY FIRST (Mandatory):**
       - Before predicting anything, visually inspect the actual photo: Palm shape (square, rectangular, broad, narrow), finger length vs palm length, thumb angle/flexibility, skin texture, line depth (deep/dark vs faint/chained), and which specific mounts are genuinely raised or flat in THIS photo.
       - Thumb Orientation Rule: Palm facing camera -> Thumb on LEFT = Right Hand (Daaya); Thumb on RIGHT = Left Hand (Baaya). Silently verify and note if swapped.
    2. **NO COPY-PASTE TEMPLATES OR FIXED AGES:**
       - Do NOT assume every hand has the exact same lines or fixed age milestones. Calculate age milestones dynamically based on where lines actually intersect, break, fork, or rise in THIS specific photo.
       - Mention ONLY lines, mounts, and marks actually visible in the uploaded photo. Do NOT invent invisible marks.
    3. **DEEP, PERSONALIZED 6-PART ANALYSIS:**
       - **Visual Proof:** First describe the exact physical shape of the palm, fingers, and main lines visible in THIS photo so the user knows you actually saw their hand.
       - **Psychology & Mind Mirror:** Deduce their real thinking style, emotional triggers, and hidden strengths strictly from their actual Head line, Heart line, and Thumb shape.
       - **Core Problem & Obstacles:** Identify the true recurring obstacle visible on THIS palm and explain its exact palmistry cause.
       - **Good vs Bad Yogas:** Separate real strong signs from weak/blocked areas visible in the photo.
       - **Timing & Remedies:** Give specific age brackets calculated from this hand's geometry and targeted remedies from Bhag 3 PDF matching ONLY the doshas actually present.

    Return strictly valid JSON in both Hindi (Devanagari) and English:
    {{
        "swabhav_vyavahar_aur_antarmann": {{
            "hindi": "Start by describing the exact physical shape of their palm, fingers, and main lines seen in THIS photo, then reveal their true psychology and hidden habits in Hindi",
            "english": "Start by describing the exact physical shape of their palm, fingers, and main lines seen in THIS photo, then reveal their true psychology in English"
        }},
        "haath_ki_pehchan_aur_tulna": {{
            "hindi": "Verification of Right Hand (karma) and Left Hand (destiny) and specific comparison of differences between both hands in Hindi",
            "english": "Verification of Right and Left hands and specific comparison in English"
        }},
        "sabse_badi_samasya_aur_bar_bar_hone_wali_rukawat": {{
            "hindi": "Specific recurring life/mental/career obstacle deduced strictly from the weak lines or mounts visible in THIS photo in Hindi",
            "english": "Specific recurring obstacle deduced strictly from THIS photo in English"
        }},
        "rekhaon_ka_sukshma_nirikshan_achha_aur_bura": {{
            "hindi": "Exact path, depth, breaks, and forks of Life, Head, Heart, Fate, and Sun lines visible in THIS photo, separating GOOD yogas from BAD doshas in Hindi",
            "english": "Exact path and marks of lines visible in THIS photo, separating GOOD yogas from BAD doshas in English"
        }},
        "sanyukt_bhavishyafal_aur_umra_ganana": {{
            "hindi": "Personalized prediction for {question_topic} with custom age milestones calculated from THIS hand's line intersections in Hindi",
            "english": "Personalized prediction with custom age milestones calculated from THIS hand in English"
        }},
        "upay_aur_karm_sudhar": {{
            "hindi": "Targeted remedies matching ONLY the specific doshas found in THIS photo in Hindi",
            "english": "Targeted remedies matching ONLY the specific doshas found in THIS photo in English"
        }}
    }}
    """
    content_parts.append(prompt)

    prediction_result = None
    last_error = ""

    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
    ]
    try:
        for m in client.models.list():
            if "pro" in m.name.lower() or "flash" in m.name.lower():
                if m.name not in preferred_models:
                    preferred_models.append(m.name)
    except Exception as e:
        last_error = f"List models error: {str(e)}"

    for round_num in range(2):
        for model_name in preferred_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        response_mime_type="application/json",
                    ),
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
@app.get("/")
def serve_website():
    return FileResponse("index.html")


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
        expiry = now + timedelta(days=3)
        cur.execute(
            "INSERT INTO memberships (phone, plan_name, expires_at, single_credits) VALUES (?, ?, ?, ?)",
            (phone, "3-Day Free Trial", expiry.isoformat(), 0)
        )
        conn.commit()
        conn.close()
        return {
            "unlocked": True,
            "plan": "3-Day Free Trial (नया यूजर ऑफर)",
            "expires_at": expiry.strftime("%d %b %Y, %I:%M %p")
        }

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
    return {
        "order_id": order["id"],
        "amount": plan["amount"],
        "key_id": RAZORPAY_KEY_ID,
        "plan_name": plan["name"]
    }


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
        cur.execute("SELECT expires_at, single_credits FROM memberships WHERE phone = ?", (req.phone.strip(),))
        row = cur.fetchone()
        old_exp = row[0] if row else now.isoformat()
        old_credits = row[1] if row else 0
        cur.execute(
            "INSERT OR REPLACE INTO memberships (phone, plan_name, expires_at, single_credits) VALUES (?, ?, ?, ?)",
            (req.phone.strip(), plan["name"], old_exp, old_credits + plan["credits"])
        )

    conn.commit()
    conn.close()
    return {"status": "unlocked", "plan": plan["name"]}
