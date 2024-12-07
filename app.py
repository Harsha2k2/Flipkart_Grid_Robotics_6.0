from flask import Flask, request, jsonify, render_template, redirect
from datetime import datetime
import pytz
import os
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
import calendar

load_dotenv()

app = Flask(__name__)

# Create a class to manage the counter and history
class AnalysisManager:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.branded_counter = 0
        self.fresh_counter = 0
        self.branded_history = []
        self.fresh_history = []
    
    def get_next(self, type):
        if type == 'branded':
            self.branded_counter += 1
            return self.branded_counter
        else:
            self.fresh_counter += 1
            return self.fresh_counter
    
    def add_to_history(self, type, data):
        if type == 'branded':
            self.branded_history.append(data)
        else:
            self.fresh_history.append(data)
    
    def get_history(self, type):
        return self.branded_history if type == 'branded' else self.fresh_history

# Initialize manager
manager = AnalysisManager()

@app.route('/reset/<type>')
def reset_counter(type):
    try:
        if type == 'all':
            manager.reset()
        elif type == 'branded':
            manager.branded_counter = 0
            manager.branded_history = []
        elif type == 'fresh':
            manager.fresh_counter = 0
            manager.fresh_history = []
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error resetting: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history/<type>')
def get_history(type):
    return jsonify(manager.get_history(type))

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def prepare_image_for_gemini(image_file):
    """
    Prepare image for Gemini API
    """
    # Read image file
    img = Image.open(image_file)
    
    # Convert to RGB if necessary
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Create byte stream
    byte_stream = io.BytesIO()
    img.save(byte_stream, format='JPEG')
    byte_stream.seek(0)
    
    return {
        "mime_type": "image/jpeg",
        "data": byte_stream.getvalue()
    }

def analyze_image_with_llm(image_file, system_prompt):
    """
    Analyze image using Gemini Pro Vision
    """
    try:
        # Prepare image for Gemini
        image_data = prepare_image_for_gemini(image_file)
        
        # Generate response from Gemini
        response = model.generate_content([
            system_prompt,
            image_data
        ])
        
        # Process the response based on prompt type
        if "packaged" in system_prompt.lower():
            # Parse packaged product response
            response_text = response.text.lower()
            return {
                'type': 'packaged',
                'brand': extract_brand(response_text),
                'expiry_date': extract_date(response_text),
                'count': extract_count(response_text)
            }
        else:
            # Parse fresh produce response
            response_text = response.text.lower()
            return {
                'type': 'fresh',
                'produce_type': extract_produce(response_text),
                'freshness_score': extract_freshness(response_text),
                'shelf_life': extract_shelf_life(response_text),
                'count': extract_count(response_text)
            }
            
    except Exception as e:
        print(f"Error in analyze_image_with_llm: {str(e)}")
        raise Exception(f"Failed to analyze image: {str(e)}")

# Helper functions to extract information from Gemini's response
def extract_brand(response_text):
    # Add logic to extract brand from response
    # This is a simple example - you might need more sophisticated parsing
    if "brand:" in response_text:
        return response_text.split("brand:")[1].split("\n")[0].strip()
    return "Not detected"

def extract_date(response_text):
    """
    Enhanced date extraction that handles various date formats
    """
    if "expiry date:" in response_text:
        date_text = response_text.split("expiry date:")[1].split("\n")[0].strip()
        
        try:
            # Try to parse full date format (YYYY-MM-DD)
            return datetime.strptime(date_text, '%Y-%m-%d').strftime('%Y-%m-%d')
        except:
            try:
                # Try to parse month-year format
                if len(date_text.split('-')) == 2:  # Format: YYYY-MM
                    year, month = map(int, date_text.split('-'))
                    # Get the last day of the month
                    last_day = calendar.monthrange(year, month)[1]
                    return f"{year}-{month:02d}-{last_day}"
                
                # Try to parse other common formats
                for fmt in ['%m/%Y', '%Y/%m', '%m-%Y', '%Y-%m', '%b %Y', '%B %Y']:
                    try:
                        date_obj = datetime.strptime(date_text, fmt)
                        year = date_obj.year
                        month = date_obj.month
                        last_day = calendar.monthrange(year, month)[1]
                        return f"{year}-{month:02d}-{last_day}"
                    except:
                        continue
                        
            except:
                pass
    
    return "Not visible"

def extract_produce(response_text):
    """
    Enhanced produce type extraction
    """
    if "produce type:" in response_text:
        produce = response_text.split("produce type:")[1].split("\n")[0].strip()
        return produce if produce and produce.lower() != "unknown" else "Not identified"
    
    # Try alternative patterns
    patterns = ["type:", "fruit:", "vegetable:", "produce:"]
    for pattern in patterns:
        if pattern in response_text.lower():
            produce = response_text.split(pattern)[1].split("\n")[0].strip()
            return produce if produce and produce.lower() != "unknown" else "Not identified"
    
    return "Not identified"

def extract_freshness(response_text):
    """
    Enhanced freshness extraction with better validation
    """
    if "freshness score:" in response_text.lower():
        try:
            # Extract the score
            score_text = response_text.lower().split("freshness score:")[1].split("\n")[0].strip()
            score = int(''.join(filter(str.isdigit, score_text)))
            
            # Validate the score
            if score < 1:
                return 1
            elif score > 10:
                return 10
            return score
        except:
            pass
    return 0

def extract_shelf_life(response_text):
    if "shelf life:" in response_text:
        try:
            days = int(response_text.split("shelf life:")[1].split("days")[0].strip())
            return max(days, 0)  # Ensure non-negative
        except:
            pass
    return 0

def extract_count(response_text):
    if "count:" in response_text:
        try:
            return int(response_text.split("count:")[1].split("\n")[0].strip())
        except:
            pass
    return 1

def validate_image(image_file):
    """
    Validate image file format and size
    """
    try:
        img = Image.open(image_file)
        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Validate image size (max 5MB)
        if image_file.tell() > 5 * 1024 * 1024:
            raise ValueError("Image size too large (max 5MB)")
            
        # Reset file pointer
        image_file.seek(0)
        return True
        
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

def process_image(image_file):
    """
    Main function to process image and extract all relevant information
    """
    # Validate image first
    validate_image(image_file)
    
    # Initial analysis to determine product type with more specific prompt
    initial_prompt = """
    Look at this image carefully and determine if it's a packaged product or fresh produce (fruit/vegetable).
    If you see any packaging, branding, or processed food items, classify it as 'packaged'.
    If you see any fruits, vegetables, or fresh produce, classify it as 'fresh'.
    
    Respond in EXACTLY this format:
    Type: [packaged/fresh]
    Count: [number of items]
    """
    
    initial_analysis = analyze_image_with_llm(image_file, initial_prompt)
    image_file.seek(0)  # Reset file pointer
    
    # Based on the type, do specific analysis
    if 'packaged' in initial_analysis.get('type', '').lower():
        return process_packaged_product(image_file, initial_analysis.get('count', 1))
    else:
        # Default to fresh produce if not clearly packaged
        return process_fresh_produce(image_file, initial_analysis.get('count', 1))

def process_packaged_product(image_file, item_count):
    """
    Process packaged products to extract brand and expiry
    """
    prompt = """
    Analyze this packaged product and provide information in the following format:
    Brand: [brand name]
    Expiry Date: [YYYY-MM-DD]
    Count: [number of items]
    
    Be precise and provide only the requested information in the specified format.
    """
    
    details = analyze_image_with_llm(image_file, prompt)
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    return {
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'brand': details.get('brand', 'Not detected'),
        'expiry_date': details.get('expiry_date', 'Not visible'),
        'count': item_count,
        'expected_life_span_days': calculate_shelf_life(details.get('expiry_date')),
        'type': 'packaged'
    }

def process_fresh_produce(image_file, item_count):
    """
    Process fresh produce to assess freshness with more detailed analysis
    """
    prompt = """
    Analyze this fresh produce (fruit or vegetable) in detail. Pay special attention to:
    - Color (looking for discoloration, browning, or dark spots)
    - Texture (looking for wrinkles, bruises, or soft spots)
    - Overall appearance (mold, decay, or other visible issues)
    
    Provide information in EXACTLY this format:
    Produce Type: [specific name of fruit/vegetable]
    Freshness Score: [score 1-10, where:
        1-2: Severely degraded/rotten
        3-4: Poor quality with visible decay
        5-6: Showing age but still edible
        7-8: Good condition
        9-10: Excellent/Peak freshness]
    Visual Issues: [list any visible problems]
    Shelf Life: [estimated remaining days]
    
    Example for bad banana:
    Produce Type: Banana
    Freshness Score: 2
    Visual Issues: Black peel, overripe, soft texture
    Shelf Life: 0

    Example for fresh apple:
    Produce Type: Apple
    Freshness Score: 9
    Visual Issues: None visible
    Shelf Life: 7
    """
    
    details = analyze_image_with_llm(image_file, prompt)
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    # Extract and validate freshness score
    freshness_score = details.get('freshness_score', 0)
    if freshness_score <= 2:
        freshness_status = "Poor/Rotten"
    elif freshness_score <= 4:
        freshness_status = "Below Average"
    elif freshness_score <= 6:
        freshness_status = "Average"
    elif freshness_score <= 8:
        freshness_status = "Good"
    else:
        freshness_status = "Excellent"
    
    # Get shelf life based on freshness score
    shelf_life = calculate_fresh_produce_shelf_life(
        details.get('produce_type', ''),
        freshness_score,
        details.get('visual_issues', '')
    )
    
    return {
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'produce': details.get('produce_type', 'Unknown produce'),
        'freshness': f"{freshness_score}/10 ({freshness_status})",
        'count': item_count,
        'expected_life_span_days': shelf_life,
        'type': 'fresh',
        'visual_issues': details.get('visual_issues', 'None reported')
    }

def calculate_fresh_produce_shelf_life(produce_type, freshness_score, visual_issues):
    """
    Calculate remaining shelf life based on produce type and condition
    """
    # Base shelf life for common produce (in days)
    base_shelf_life = {
        'banana': 7,
        'apple': 14,
        'orange': 14,
        'tomato': 7,
        'potato': 21,
        'onion': 30,
        'carrot': 21,
        'lettuce': 7,
        'cucumber': 7,
        'grape': 7,
        # Add more produce types as needed
    }
    
    produce_type = produce_type.lower()
    base_days = base_shelf_life.get(produce_type, 7)  # Default to 7 if produce type unknown
    
    # Adjust based on freshness score
    if freshness_score <= 2:
        return 0  # Already spoiled
    elif freshness_score <= 4:
        return max(1, int(base_days * 0.2))  # 20% of base shelf life
    elif freshness_score <= 6:
        return max(2, int(base_days * 0.5))  # 50% of base shelf life
    elif freshness_score <= 8:
        return max(3, int(base_days * 0.7))  # 70% of base shelf life
    else:
        return base_days  # Full shelf life

def calculate_shelf_life(expiry_date):
    """
    Calculate remaining shelf life in days with improved date handling
    """
    if not expiry_date or expiry_date == 'Not visible':
        return 0
    try:
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        today = datetime.now()
        remaining_days = (expiry - today).days
        return max(0, remaining_days)
    except:
        return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze/<type>')
def analyze_page(type):
    if type not in ['branded', 'fresh']:
        return redirect('/')
    return render_template('analyze.html', type=type)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        product_type = request.form.get('product_type')
        
        # Process the image based on the selected product type
        if product_type == 'branded':
            result = process_packaged_product(image_file, 1)
        else:
            result = process_fresh_produce(image_file, 1)
        
        # Get the next counter value and create response
        serial_number = manager.get_next(product_type)
        response = {
            'serial_number': serial_number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'brand': result.get('brand', 'Not detected') if product_type == 'branded' else result.get('produce', 'Unknown produce'),
            'expiry_date': result.get('expiry_date', 'Not available') if product_type == 'branded' else result.get('freshness', 'Unknown'),
            'count': result.get('count', 1),
            'expected_life_span_days': result.get('expected_life_span_days', 0)
        }
        
        # Add to history
        manager.add_to_history(product_type, response)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 