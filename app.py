import logging
from typing import Dict, Any, List, Optional
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import uvicorn
import os
from openai import OpenAI
import json
import re


logger = logging.getLogger(__name__)

# Initialize the OpenRouter client
# Replace 'your_openrouter_api_key_here' with your actual key, or set it via environment variables
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "your_openrouter_api_key_here"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


async def post_process_with_llm(raw_ocr_text: str) -> str:
    """
    Sends raw OCR text string data to OpenRouter (openai/gpt-oss-120b:free).
    Streams the deep thinking reasoning trace into the server logs for inspection,
    strips away any markdown metadata containers, and validates clean JSON objects.
    """
    if not raw_ocr_text.strip():
        return json.dumps({"error": "Empty input string supplied"}, indent=2)

    detailed_prompt = (
        "TASK INSTRUCTION:\n"
        "You are an enterprise financial extraction parsing processor. Look at the messy raw OCR text layout below "
        "extracted from a Tax Invoice. Clean up formatting defects, isolate fields, restore space breaks between word joins, "
        "and return a highly accurate, structured JSON object structure matching the data schema.\n\n"
        "EXPECTED DATA SHAPE FIELDS:\n"
        "- SellerDetails: { CompanyName, Address, PAN, Phone, Web, Email, Tel }\n"
        "- InvoiceDetails: { InvoiceNo, GSTNumber, InvoiceDate, ChallanNo, ChallanDate, EWayBillNo, TransportName, TransportID }\n"
        "- CustomerDetails: { ClientName, Address, Phone, GSTIN, PlaceOfSupply }\n"
        "- LineItems: Array of objects [ { SrNo, NameOfProduct, HSN_SAC, Qty, Rate, TaxableValue, IGST_Percentage, IGST_Amount, Total } ]\n"
        "- FinancialSummary: { TotalQty, TotalTaxableAmount, TotalIGSTAmount, TotalAmountAfterTax, TotalAmountInWords }\n"
        "- SettlementDetails: { BankName, Branch, AccountNumber, IFSC, UPI_ID, PayUsingUPI }\n"
        "- Compliance: { TermsAndConditions: [], Notes: '' }\n\n"
        "OUTPUT RULE CONSTRAINTS:\n"
        "- Fix text joins like 'GUJARATFREIGHTTOOLS' to 'GUJARAT FREIGHT TOOLS' and 'NameofProduct' to 'Name of Product'.\n"
        "- Output your entire output final payload wrapped inside a standard clean text container. No preamble, conversational notes, or post-scripts allowed.\n\n"
        f"RAW OCR INPUT STRING:\n\"\"\"\n{raw_ocr_text}\n\"\"\""
        "CRITICAL EXTRACTION RULES:\n"
"- NEVER invent, infer, correct, normalize, or guess values.\n"
"- If text is unclear, preserve the OCR value exactly.\n"
"- Do not round numbers.\n"
"- Do not fix city names, company names, bank names, GSTIN, PAN, IFSC, addresses, or totals.\n"
"- Extract only what appears in the OCR text.\n"
"- If a field is missing, return an empty string.\n"
"- Every numeric value must exactly match the OCR source.\n"
"- Maintain original invoice amounts without arithmetic adjustments.\n\n"
    )

    try:
        logger.info("Initializing streaming token channels with OpenRouter endpoint...")

        response_stream = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigid data processing utility that outputs raw JSON objects. "
                        "Never speak to the user or include introductory paragraphs."
                    ),
                },
                {
                    "role": "user",
                    "content": detailed_prompt,
                },
            ],
            extra_body={
                "reasoning": {
                    "enabled": True,
                    "effort": "low"
                }
            },
            temperature=0,
            stream=True,
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "OCR Invoice Processor",
            },
        )

        full_llm_payload_buffer = []
        reasoning_buffer = []

        print("\n🧠 [LLM REASONING TRACE START] ⬇️\n")

        for chunk in response_stream:
            try:
                if not chunk.choices:
                    continue


                delta = chunk.choices[0].delta

                # Reasoning field
                if hasattr(delta, "reasoning") and delta.reasoning:
                    print(delta.reasoning, end="", flush=True)
                    reasoning_buffer.append(delta.reasoning)

                # Reasoning content field
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    print(delta.reasoning_content, end="", flush=True)
                    reasoning_buffer.append(delta.reasoning_content)


                # Content field
                if hasattr(delta, "content") and delta.content:
                    print(delta.content, end="", flush=True)
                    full_llm_payload_buffer.append(delta.content)

            except Exception as stream_chunk_error:
                logger.warning(
                    f"Chunk processing error: {stream_chunk_error}"
                )

        print("\n\n🧠 [LLM REASONING TRACE END] ⬆️\n")

        if reasoning_buffer:
            print("\n==============================")
            print("FULL REASONING OUTPUT")
            print("==============================")
            print("".join(reasoning_buffer))
            print("==============================\n")
        else:
            print(
                "\n⚠️ No reasoning tokens were returned by the provider/model.\n"
            )

        compiled_output_text = "".join(full_llm_payload_buffer).strip()

        if "```" in compiled_output_text:
            json_match = re.search(
                r"```(?:json)?\s*([\s\S]*?)\s*```",
                compiled_output_text,
                re.IGNORECASE,
            )

            if json_match:
                compiled_output_text = json_match.group(1).strip()

        try:
            parsed_json_check = json.loads(compiled_output_text)

            return json.dumps(
                parsed_json_check,
                indent=2,
                ensure_ascii=False,
            )

        except json.JSONDecodeError:
            logger.warning(
                "LLM string returned invalid structured formatting. Attempting regex extraction fallback..."
            )

            raw_braces_match = re.search(
                r"(\{[\s\S]*\})",
                compiled_output_text,
            )

            if raw_braces_match:
                try:
                    parsed_json_check = json.loads(
                        raw_braces_match.group(1)
                    )

                    return json.dumps(
                        parsed_json_check,
                        indent=2,
                        ensure_ascii=False,
                    )

                except json.JSONDecodeError:
                    pass

            return json.dumps(
                {
                    "raw_output": compiled_output_text,
                    "parsing_error": "Structural layout anomaly detected",
                },
                indent=2,
                ensure_ascii=False,
            )

    except Exception as llm_error:
        logger.error(
            f"Critical execution error interfacing OpenRouter endpoints: {llm_error}",
            exc_info=True,
        )

        fallback_json = {
            "success": False,
            "error_log": str(llm_error),
            "fallback_raw_data": raw_ocr_text,
        }

        return json.dumps(
            fallback_json,
            indent=2,
            ensure_ascii=False,
        )







# Setup aggressive performance logging configurations
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rapidocr_api")

# Initialize highly optimized FastAPI core instance
app = FastAPI(
    title="RapidOCR Enterprise Microservice",
    description="Blazing fast, memory-optimized OCR service powered by ONNX Runtime.",
    version="2.0.0"
)

# Enable modern CORS protocols for direct Node.js or browser client ingestion
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Harden this array explicitly before rolling into cloud production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RapidOCR globally during execution spin-up to eliminate runtime model reloading latency
try:
    # Defaulting setup handles standard languages elegantly with CPU multi-threading
    engine = RapidOCR()
    logger.info("ONNX-Runtime inference engine safely mapped to memory pool.")
except Exception as init_error:
    logger.error(f"Critical operational fault during core engine initialization: {init_error}")
    raise init_error


def calculate_execution_time(elapse_data: Any) -> float:
    """Safely calculates total operational latency across varying library iterations."""
    if isinstance(elapse_data, list):
        return float(sum(elapse_data))
    return float(elapse_data) if elapse_data else 0.0


def format_structural_data(raw_results: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Converts multi-dimensional nested array schemas directly into lightning-fast JSON objects."""
    if not raw_results:
        return []
    
    structured_list = []
    for polygon, text_string, confidence_score in raw_results:
        try:
            structured_list.append({
                "text": str(text_string).strip(),
                "confidence": round(float(confidence_score), 4),
                "box": [[int(coordinate[0]), int(coordinate[1])] for coordinate in polygon]
            })
        except (ValueError, IndexError, TypeError):
            continue  # Silently skip corrupted spatial fragments to maximize speed
    return structured_list


def construct_natural_text(raw_results: Optional[List[Any]]) -> str:
    """
    Advanced layout sorting for tables and multi-column documents.
    Groups text segments into unified lines using a vertical pixel threshold.
    """
    if not raw_results:
        return ""
    
    try:
        # Extract operational elements into a workable dictionary list
        valid_items = []
        for polygon, text_string, _ in raw_results:
            if not text_string:
                continue
            # Extract spatial boundary bounds
            x_coordinates = [pt[0] for pt in polygon]
            y_coordinates = [pt[1] for pt in polygon]
            
            min_x, max_x = min(x_coordinates), max(x_coordinates)
            min_y, max_y = min(y_coordinates), max(y_coordinates)
            center_y = (min_y + max_y) / 2
            
            valid_items.append({
                "text": str(text_string).strip(),
                "min_x": min_x,
                "center_y": center_y,
                "height": max_y - min_y
            })

        if not valid_items:
            return ""

        # Sort all elements from top to bottom by center_y first
        valid_items.sort(key=lambda item: item["center_y"])

        # Group text items into rows using a dynamic vertical overlapping threshold
        rows = []
        current_row = [valid_items[0]]
        
        for item in valid_items[1:]:
            # Determine threshold based on average text segment height
            threshold = min(current_row[-1]["height"], item["height"]) * 0.5
            
            # If the block falls within the row height boundary, add to current row
            if abs(item["center_y"] - current_row[-1]["center_y"]) <= threshold:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
        rows.append(current_row)

        # Sort each individual row horizontally (left-to-right) and join words with space
        final_lines = []
        for row in rows:
            row.sort(key=lambda item: item["min_x"])
            # Join items with spaces to separate tabular column data cleanly
            line_text = "   ".join([item["text"] for item in row])
            final_lines.append(line_text)

        return "\n".join(final_lines)

    except Exception as sorting_error:
        logger.warning(f"Line grouping threshold failed, falling back to primitive array sequence. Error: {sorting_error}")
        return "\n".join([str(element[1]).strip() for element in raw_results if element and len(element) > 1])


async def process_and_decode_upload(uploaded_file: UploadFile) -> tuple:
    """Reusable pipeline component optimizing parallel payload validation and OpenCV ingestion."""
    if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Incompatible file payload: '{uploaded_file.content_type}'. Only images are supported."
        )

    binary_stream = await uploaded_file.read()
    if not binary_stream:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Zero-byte image payload submitted.")

    byte_buffer = np.frombuffer(binary_stream, np.uint8)
    decoded_matrix = cv2.imdecode(byte_buffer, cv2.IMREAD_COLOR)
    
    if decoded_matrix is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="OpenCV core parser failed to compile target image binary layout."
        )
        
    return decoded_matrix


@app.get("/health", status_code=status.HTTP_200_OK)
async def service_health_check() -> Dict[str, str]:
    """Endpoint tracking operational infrastructure availability."""
    return {"status": "operational", "runtime": "ONNX_Inference_Ready"}


@app.post("/ocr", status_code=status.HTTP_200_OK)
async def extract_text_only(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    HIGH SPEED ROUTE: Drops resource-heavy pixel coordinate structures.
    Directly outputs pure structured data processed through openrouter LLM parsing.
    """
    try:
        image_matrix = await process_and_decode_upload(file)
        ocr_output, processing_elapse = engine(image_matrix)
        
        # 1. Run our internal line binning and spatial text repair logic 
        structured_raw_text = construct_natural_text(ocr_output)
        
        # 2. Intercept text stream and pipeline it through the OpenRouter free LLM instance
        logger.info("Forwarding text array stream to openrouter (openai/gpt-oss-120b:free) for structural normalization...")
        final_processed_output = await post_process_with_llm(structured_raw_text)
        
        return {
            "success": True,
            "inference_time_seconds": calculate_execution_time(processing_elapse),
            "processed_output": final_processed_output,
            "raw_ocr_text": structured_raw_text
        }
    except HTTPException as handled_exception:
        raise handled_exception
    except Exception as unhandled_error:
        logger.error(f"Execution error on route '/ocr': {unhandled_error}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OCR parsing runtime exception.")
    finally:
        await file.close()



@app.post("/ocr/boxes", status_code=status.HTTP_200_OK)
async def extract_detailed_structural_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    DETAILED ROUTE: Extracts complete text layers with precise analytical metrics.
    Returns pixel coordinates and detection engine confidence scoring indices.
    """
    try:
        image_matrix = await process_and_decode_upload(file)
        frame_height, frame_width, _ = image_matrix.shape
        
        ocr_output, processing_elapse = engine(image_matrix)
        compiled_boxes = format_structural_data(ocr_output)
        
        return {
            "success": True,
            "metadata": {
                "filename": file.filename,
                "width": frame_width,
                "height": frame_height,
                "inference_time_seconds": calculate_execution_time(processing_elapse),
                "total_segments_discovered": len(compiled_boxes)
            },
            "data": compiled_boxes
        }
    except HTTPException as handled_exception:
        raise handled_exception
    except Exception as unhandled_error:
        logger.error(f"Execution error on route '/ocr/boxes': {unhandled_error}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Detailed data runtime exception.")
    finally:
        await file.close()


if __name__ == "__main__":
    # Optimized deployment parameters
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Permits container virtualization exposures
        port=8000,
        workers=1,       # RapidOCR ONNX handles internal threads; keep app-workers at 1 to prevent thread collisions
        reload=False     # Turn off template tracking to free CPU cycles
    )
