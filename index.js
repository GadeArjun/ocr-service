import fs from 'node:fs';
import path from 'node:path';

// CONFIGURATION: Target execution configuration
const IMAGE_PATH = './image.png'; 
const BASE_API_URL = 'https://orange-palm-tree-4jwp9w6x75743qvg5-8000.app.github.dev';

/**
 * Generates web-standard FormData with explicit payload attachments
 */
function createOcrPayload(imagePath) {
    const fileBuffer = fs.readFileSync(imagePath);
    const ext = path.extname(imagePath).toLowerCase();
    
    let mimeType = 'image/png';
    if (ext === '.jpg' || ext === '.jpeg') mimeType = 'image/jpeg';
    if (ext === '.webp') mimeType = 'image/webp';

    const blob = new Blob([fileBuffer], { type: mimeType });
    const file = new File([blob], path.basename(imagePath), { type: mimeType });
    
    const formData = new FormData();
    formData.append('file', file);
    return formData;
}

async function testOCR() {
    try {
        // 1. Validation guard clause
        if (!fs.existsSync(IMAGE_PATH)) {
            console.error(`❌ File System Error: Target image missing at "${IMAGE_PATH}"`);
            return;
        }

        console.log(`⏳ Loading asset file: "${IMAGE_PATH}"...`);

        // ==========================================
        // ROUTE 1 TESTING: Pure Text Ingestion (/ocr)
        // ==========================================
        console.log(`\n🚀 [1/2] Fetching pure text layout from: ${BASE_API_URL}/ocr`);
        const textPayload = createOcrPayload(IMAGE_PATH);
        const textResponse = await fetch(`${BASE_API_URL}/ocr`, {
            method: 'POST',
            body: textPayload,
        });

        const textData = await textResponse.json();

        console.log(JSON.stringify(textData, null, 2));

        if (!textResponse.ok) {
            console.error(`❌ Text Route Error (${textResponse.status}):`, textData);
        } else {
            console.log('✅ Text Extraction Complete!');
            console.log('--------------------------------------------------');
            console.log(`⏱️  Inference Time: ${textData.inference_time_seconds.toFixed(4)} seconds`);
            console.log('📝 EXTRACTED TEXT (Natural Layout Order):');
            console.log(textData.text ? textData.text : '   [No text detected]');
            console.log('--------------------------------------------------');
        }

        // ==========================================
        // ROUTE 2 TESTING: Analytical Metadata (/ocr/boxes)
        // ==========================================
        console.log(`\n🚀 [2/2] Fetching spatial coordinate layout from: ${BASE_API_URL}/ocr/boxes`);
        const boxPayload = createOcrPayload(IMAGE_PATH);
        const boxResponse = await fetch(`${BASE_API_URL}/ocr/boxes`, {
            method: 'POST',
            body: boxPayload,
        });

        const boxData = await boxResponse.json();

        if (!boxResponse.ok) {
            console.error(`❌ Box Route Error (${boxResponse.status}):`, boxData);
            return;
        }

        console.log('✅ Structural Coordinates Complete!');
        console.log('--------------------------------------------------');
        console.log('📊 METADATA:');
        console.log(`   File Name:       ${boxData.metadata.filename}`);
        console.log(`   Resolution:      ${boxData.metadata.width}x${boxData.metadata.height}`);
        console.log(`   Inference Time:  ${boxData.metadata.inference_time_seconds.toFixed(4)} seconds`);
        console.log(`   Segments Found:  ${boxData.metadata.total_segments_discovered}`);
        console.log('--------------------------------------------------');
        
        console.log('🗺️  DETAILED SEGMENT SEGREGATION:');
        if (boxData.data.length === 0) {
            console.log('   [No segmented vectors mapped]');
        } else {
            boxData.data.forEach((segment, index) => {
                console.log(`   [Line ${index + 1}] Text: "${segment.text}"`);
                console.log(`          Confidence Score: ${(segment.confidence * 100).toFixed(2)}%`);
                console.log(`          Bounding Polygon: ${JSON.stringify(segment.box)}`);
            });
        }
        console.log('--------------------------------------------------\n');

    } catch (networkError) {
        console.error('\n❌ Execution Exception encountered during API testing:');
        console.error(`   Message: ${networkError.message}`);
    }
}

// Fire the validation sequence
testOCR();
