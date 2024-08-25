import traceback
from typing import Any, Optional, Tuple

import cv2
import easyocr
import numpy as np
from mrz.checker.td3 import TD3CodeChecker


def calculate_check_digit(data: str) -> Optional[str]:
	weight = [7, 3, 1]
	total = sum(
		(int(char) if char.isdigit() else (ord(char) - ord("A") + 10 if char.isalpha() else 0)) * weight[i % 3]
		for i, char in enumerate(data)
		if char != "<"
	)
	return str(total % 10)


def validate_mrz_field(mrz_field: str) -> bool:
	data, expected_check_digit = mrz_field[:-1], mrz_field[-1]
	calculated_check_digit = calculate_check_digit(data)
	return calculated_check_digit == expected_check_digit if calculated_check_digit else False


def reformat_date(date_str: str) -> str:
	if len(date_str) != 6:
		raise ValueError("Date string must be in YYMMDD format")
	day, month, year = date_str[4:], date_str[2:4], date_str[:2]
	year = f"{'19' if int(year) > 50 else '20'}{year}"
	return f"{day}.{month}.{year}"


def preprocess_image(image: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
	_, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return binary


def crop_mrz(image: np.ndarray) -> np.ndarray:
	height, width = image.shape[:2]
	return image[int(height * 0.8) :, :]


def detect_and_crop_face(image: np.ndarray, padding_percent: float = 0.2) -> Optional[np.ndarray]:
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

	if len(faces) > 0:
		x, y, w, h = faces[0]
		padding = int(w * padding_percent)
		x_start, y_start = max(x - padding, 0), max(y - padding, 0)
		x_end, y_end = min(x + w + padding, image.shape[1]), min(y + h + padding, image.shape[0])
		return image[y_start:y_end, x_start:x_end]
	return None


def read_mrz_with_easyocr(image: np.ndarray) -> Optional[str]:
	reader = easyocr.Reader(["en"])
	results = reader.readtext(image, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
	mrz_text = "".join(results).replace(" ", "")
	return mrz_text if mrz_text else None


def improve_mrz_accuracy(image: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
	preprocessed_image = preprocess_image(image)
	cropped_mrz = crop_mrz(preprocessed_image)

	for scale_percent in [100, 150, 200]:
		resized_image = cv2.resize(cropped_mrz, None, fx=scale_percent / 100, fy=scale_percent / 100)
		mrz_text = read_mrz_with_easyocr(resized_image)
		if mrz_text:
			face_img = detect_and_crop_face(image)
			return mrz_text, face_img, cropped_mrz

	return None, None, None


def parse_mrz_fields(mrz_text: str) -> dict[str, dict[str, Any]]:
	try:
		td3_check = TD3CodeChecker(mrz_text, check_expiry=True)
	except Exception as e:
		print(f"Error parsing MRZ: {e}")
		return {}

	fields = td3_check.fields()
	invalid_fields = [field[0] for field in td3_check.report.falses] if td3_check.report.falses else []

	def check_falses(key_words):
		return not any(key_words in field for field in invalid_fields)

	return {
		"full_mrz": {"value": mrz_text, "status": check_falses("final hash")},
		"mrz_birth_date": {"value": reformat_date(fields.birth_date), "status": check_falses("birth")},
		"mrz_cd_birth_date": {"value": fields.birth_date_hash, "status": check_falses("birth date hash")},
		"mrz_cd_composite": {"value": fields.final_hash, "status": True},
		"mrz_cd_expiry_date": {"value": fields.expiry_date_hash, "status": check_falses("expiry date hash")},
		"mrz_cd_number": {"value": fields.document_number_hash, "status": check_falses("document number hash")},
		"mrz_cd_opt_data_2": {"value": fields.optional_data_hash, "status": check_falses("optional data hash")},
		"mrz_doc_type_code": {"value": fields.document_type, "status": check_falses("document type")},
		"mrz_expiry_date": {"value": reformat_date(fields.expiry_date), "status": check_falses("expiry date")},
		"mrz_gender": {"value": fields.sex, "status": check_falses("sex")},
		"mrz_issuer": {"value": fields.country, "status": check_falses("nationality")},
		"mrz_last_name": {"value": fields.surname, "status": True},
		"mrz_line1": {"value": mrz_text.split("\n")[0], "status": check_falses("final hash")},
		"mrz_line2": {"value": mrz_text.split("\n")[1], "status": check_falses("final hash")},
		"mrz_name": {"value": fields.name, "status": True},
		"mrz_nationality": {"value": fields.nationality, "status": check_falses("nationality")},
		"mrz_number": {"value": fields.document_number, "status": check_falses("document number")},
		"mrz_opt_data_2": {"value": fields.optional_data, "status": check_falses("optional data")},
	}


def get_info(
	img_path: str,
) -> Tuple[dict[str, dict[str, Any]] | dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
	try:
		image = cv2.imread(img_path, cv2.IMREAD_COLOR)
		if image is None:
			return {"error": "Could not read image, please check the path."}, None, None

		mrz_text, face_img, mrz_code_img = improve_mrz_accuracy(image)
		if mrz_text:
			fields_dict = parse_mrz_fields(mrz_text)
			return fields_dict, face_img, mrz_code_img
		else:
			return {"error": "Couldn't extract data from image. Try a clearer image."}, None, None
	except Exception as e:
		print(traceback.format_exc())
		return {"error": f"Failed to process image: {str(e)}"}, None, None
