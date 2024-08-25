import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.passport_logic import get_info

app = FastAPI()

# Configure CORS
app.add_middleware(
		CORSMiddleware,  # type: ignore
		allow_origins=["*"],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

os.getcwd()

app.mount("/static", StaticFiles(directory=f"{os.getcwd()}/app/static"), name="static")


@app.get("/")
async def read_index():
	return FileResponse("app/static/index.html")


def encode_image_to_base64(image: Optional[cv2.Mat]) -> Optional[str]:
	if image is None:
		return None
	_, buffer = cv2.imencode(".jpg", image)
	return base64.b64encode(buffer).decode("utf-8")


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
	try:
		file_path = UPLOAD_DIR / file.filename
		content = await file.read()
		file_path.write_bytes(content)

		extracted_data, face_img, mrz_code_img = get_info(str(file_path))

		if isinstance(extracted_data, dict) and "error" in extracted_data:
			raise HTTPException(status_code=400, detail=extracted_data["error"])

		response_data: Dict[str, Any] = {
				"data"      : extracted_data,
				"face_image": encode_image_to_base64(face_img),
				"mrz_image" : encode_image_to_base64(mrz_code_img),
		}

		file_path.unlink()  # Remove the temporary file

		return JSONResponse(content=response_data)

	except HTTPException as http_ex:
		return JSONResponse(status_code=http_ex.status_code, content={"error": http_ex.detail})
	except Exception as e:
		return JSONResponse(status_code=500, content={"error": f"Failed to process the image: {str(e)}"})


if __name__ == "__main__":
	uvicorn.run(
			app,
			host="0.0.0.0",
			port=443,
			ssl_keyfile="ip_address.key",
			ssl_certfile="ip_address.crt",
	)
