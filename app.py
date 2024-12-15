import tensorflow as tf
import torch
from flask import Flask, request, jsonify, render_template, redirect
from datetime import datetime
import pytz
import os
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
import calendar
from flask_sqlalchemy import SQLAlchemy

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///product_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

expiry_model = tf.keras.models.load_model('models/expiry/model_expiry.h5')
freshness_model = torch.load('models/freshness/best_fruit_freshness_model.pth')

# Define database models
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    serial_number = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    product_type = db.Column(db.String(20), nullable=False)  # 'branded', 'fresh', or 'all'
    name = db.Column(db.String(100))  # brand name or produce type
    expiry_date = db.Column(db.String(20))
    count = db.Column(db.Integer)
    expected_life_span_days = db.Column(db.Integer)
    freshness = db.Column(db.String(50))
    
    def to_dict(self):
        return {
            'serial_number': self.serial_number,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'product_type': self.product_type,
            'name': self.name,
            'expiry_date': self.expiry_date,
            'count': self.count,
            'expected_life_span_days': self.expected_life_span_days,
            'freshness': self.freshness
        }

# Create database tables
with app.app_context():
    db.create_all()

# Create a class to manage the counter and history
class AnalysisManager:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.branded_counter = 0
        self.fresh_counter = 0
        self.all_counter = 0  # New counter for all products
        self.branded_history = []
        self.fresh_history = []
        self.all_history = []  # New history for all products
    
    def get_next(self, type):
        if type == 'branded':
            self.branded_counter += 1
            return self.branded_counter
        elif type == 'fresh':
            self.fresh_counter += 1
            return self.fresh_counter
        else:  # 'all'
            self.all_counter += 1
            return self.all_counter
    
    def add_to_history(self, type, data):
        if type == 'branded':
            self.branded_history.append(data)
        elif type == 'fresh':
            self.fresh_history.append(data)
        else:  # 'all'
            self.all_history.append(data)
    
    def get_history(self, type):
        if type == 'branded':
            return self.branded_history
        elif type == 'fresh':
            return self.fresh_history
        return self.all_history  # 'all'

# Initialize manager
manager = AnalysisManager()

@app.route('/reset/<type>')
def reset_counter(type):
    try:
        if type == 'all':
            manager.reset()
            db.session.query(Analysis).delete()
        elif type == 'branded':
            manager.branded_counter = 0
            manager.branded_history = []
            db.session.query(Analysis).filter_by(product_type='branded').delete()
        elif type == 'fresh':
            manager.fresh_counter = 0
            manager.fresh_history = []
            db.session.query(Analysis).filter_by(product_type='fresh').delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error resetting: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history/<type>')
def get_history(type):
    try:
        if type == 'all':
            analyses = Analysis.query.order_by(Analysis.timestamp.desc()).all()
        else:
            analyses = Analysis.query.filter_by(product_type=type).order_by(Analysis.timestamp.desc()).all()
        return jsonify([analysis.to_dict() for analysis in analyses])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
print(f"API Key: {GOOGLE_API_KEY[:5]}...") # Only print first 5 chars for security
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
    """
    Enhanced brand extraction with better validation
    """
    if "brand:" in response_text.lower():
        try:
            brand = response_text.lower().split("brand:")[1].split("\n")[0].strip()
            # Remove common non-brand text
            brand = brand.replace('not detected', '').replace('unknown', '').strip()
            return brand.title() if brand else "Not detected"
        except:
            pass
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
    """
    Enhanced count extraction that ensures proper counting with validation
    """
    if "count:" in response_text.lower():
        try:
            count_text = response_text.lower().split("count:")[1].split("\n")[0].strip()
            # Extract only numbers from the text
            count = int(''.join(filter(str.isdigit, count_text)))
            
            # Add reasonable limits
            if count < 1:
                return 1
            elif count > 20:  # Adjust this maximum limit based on your needs
                return 1
                
            return count
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
    
    IMPORTANT: Count ALL items visible in the image, even if:
    - They are the same type (e.g., 5 bananas should be counted as 5)
    - They are partially visible
    - They are in different states of freshness
    
    Respond in EXACTLY this format:
    Type: [packaged/fresh]
    Count: [exact number of items visible]
    """
    
    initial_analysis = analyze_image_with_llm(image_file, initial_prompt)
    image_file.seek(0)  # Reset file pointer
    
    # Based on the type, do specific analysis
    if 'packaged' in initial_analysis.get('type', '').lower():
        return process_packaged_product(image_file, initial_analysis.get('count', 1))
    else:
        # Default to fresh produce if not clearly packaged
        return process_fresh_produce(image_file, initial_analysis.get('count', 1))

def extract_mrp(response_text):
    """
    Extract MRP from the response text
    """
    if "mrp:" in response_text.lower():
        try:
            mrp_text = response_text.lower().split("mrp:")[1].split("\n")[0].strip()
            # Remove currency symbols and 'rs' text if present
            mrp_text = mrp_text.replace('₹', '').replace('rs', '').replace('rs.', '').strip()
            # Extract numbers including decimals
            mrp = float(''.join(c for c in mrp_text if c.isdigit() or c == '.'))
            return f"₹{mrp:.2f}"
        except:
            pass
    return "Not visible"

def process_packaged_product(image_file, item_count):
    """
    Process packaged products to extract brand and expiry
    """
    prompt = """
    Analyze this packaged product and provide information in the following format.
    
    COUNTING RULES:
    - Count each DISTINCT physical package/item visible in the image
    - Do NOT count the same item multiple times
    - Do NOT count logos or brand images as separate items
    - Only count actual physical products
    
    IMPORTANT: Look carefully for:
    - Brand name on the package
    - MRP (Maximum Retail Price) usually marked as MRP ₹XX or Rs.XX
    
    Provide information in EXACTLY this format:
    Brand: [exact brand name visible on the package]
    Expiry Date: [YYYY-MM-DD]
    MRP: [price in ₹ or Rs. if visible]
    Count: [number of distinct physical packages/items]
    
    Be precise and provide only the requested information in the specified format.
    """
    
    details = analyze_image_with_llm(image_file, prompt)
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    # Add validation for count
    count = details.get('count', 1)
    if count > 20:
        count = 1
    
    return {
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'brand': details.get('brand', 'Not detected'),
        'expiry_date': details.get('expiry_date', 'Not visible'),
        'mrp': extract_mrp(details.get('response_text', '')),
        'count': count,
        'expected_life_span_days': calculate_shelf_life(details.get('expiry_date')),
        'type': 'branded'
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
    
    IMPORTANT: Count ALL items visible in the image, even if:
    - They are the same type (e.g., 5 bananas should be counted as 5)
    - They are partially visible
    - They are in different states of freshness
    
    Provide information in EXACTLY this format:
    Produce Type: [specific name of fruit/vegetable]
    Count: [exact number of items visible]
    Freshness Score: [score 1-10, where:
        1-2: Severely degraded/rotten
        3-4: Poor quality with visible decay
        5-6: Showing age but still edible
        7-8: Good condition
        9-10: Excellent/Peak freshness]
    Visual Issues: [list any visible problems]
    Shelf Life: [estimated remaining days]
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
        'count': details.get('count', 1),
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
    if type not in ['branded', 'fresh', 'all']:
        return redirect('/')
    return render_template('analyze.html', type=type)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        product_type = request.form.get('product_type')
        
        # Process the image
        if product_type == 'all':
            result = process_image(image_file)
            detected_type = result.get('type', 'unknown')
            if detected_type == 'packaged':  # Convert 'packaged' to 'branded'
                detected_type = 'branded'
        else:
            if product_type == 'branded':
                result = process_packaged_product(image_file, 1)
            else:
                result = process_fresh_produce(image_file, 1)
            detected_type = product_type
        
        # Get the next counter value
        serial_number = manager.get_next(product_type)
        
        # Create unified response format
        response = {
            'serial_number': serial_number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'product_type': detected_type,
            'name': result.get('brand' if detected_type == 'branded' else 'produce', 'Unknown'),
            'expiry_date': result.get('expiry_date', 'N/A') if detected_type == 'branded' else 'N/A',
            'count': result.get('count', 1),
            'expected_life_span_days': result.get('expected_life_span_days', 0),
            'freshness': result.get('freshness', 'N/A') if detected_type == 'fresh' else 'N/A'
        }
        
        # Save to database
        analysis = Analysis(
            serial_number=serial_number,
            product_type=detected_type,
            name=response['name'],
            expiry_date=response['expiry_date'],
            count=response['count'],
            expected_life_span_days=response['expected_life_span_days'],
            freshness=response['freshness']
        )
        db.session.add(analysis)
        db.session.commit()
        
        # Add to history
        manager.add_to_history(product_type, response)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

@app.route('/dashboard')
def dashboard():
    try:
        page = request.args.get('page', 1, type=int)
        sort_order = request.args.get('sort', 'desc')
        per_page = 10
        
        # Get all scans ordered by timestamp ascending first to assign sequential IDs
        all_scans = Analysis.query.order_by(Analysis.timestamp.asc()).all()
        
        # Assign sequential IDs starting from 1
        for index, scan in enumerate(all_scans, 1):
            scan.sequential_id = index
        
        # Now sort according to user preference
        if sort_order == 'desc':
            all_scans.reverse()
            
        total_scans = len(all_scans)
        
        # Rest of the pagination logic...
        total_pages = (total_scans + per_page - 1) // per_page
        
        if page < 1:
            page = 1
        elif page > total_pages and total_pages > 0:
            page = total_pages
            
        start = (page - 1) * per_page
        end = min(start + per_page, total_scans)
        paginated_scans = all_scans[start:end]
        
        # Calculate statistics...
        stats = {
            'total_products': total_scans,
            'branded_products': len([s for s in all_scans if s.product_type == 'branded']),
            'fresh_products': len([s for s in all_scans if s.product_type == 'fresh']),
            'total_items': sum(scan.count for scan in all_scans),
            'todays_branded_scans': len([s for s in all_scans if s.product_type == 'branded' and s.timestamp.date() == datetime.now().date()]),
            'todays_fresh_scans': len([s for s in all_scans if s.product_type == 'fresh' and s.timestamp.date() == datetime.now().date()])
        }
        
        return render_template('dashboard.html', 
                             stats=stats,
                             recent_scans=paginated_scans,
                             current_page=page,
                             total_pages=max(1, total_pages),
                             sort_order=sort_order)
    except Exception as e:
        return f"Error loading dashboard: {str(e)}", 500

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=8080, debug=debug_mode) 
