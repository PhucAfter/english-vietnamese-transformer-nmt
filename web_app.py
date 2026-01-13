# web_app.py
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import hàm dịch của bạn
# Bạn sẽ tạo hàm translate_text trong translate.py ở bước 3
from translate import translate_text

app = FastAPI(title="EN → VI Translator")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Static + templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/translate")
async def api_translate(request: Request):
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "Vui lòng nhập tiếng Anh để dịch."}, status_code=400)

    try:
        vi = translate_text(text)
        return {"ok": True, "translation": vi}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Lỗi dịch: {str(e)}"}, status_code=500)
