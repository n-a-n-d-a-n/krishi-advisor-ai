# app.py — KrishiAdvisor V2
# Multi-page Flask application for Indian farmers

import os, json, hashlib, datetime
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash, g)
from database import get_db, init_db, seed_price_history
from ai_engine import full_analysis, predict_prices

app = Flask(__name__)
app.secret_key = 'krishi-v2-secret-2024-maharashtra'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Initialize DB on startup (required for Vercel)
init_db()
seed_price_history()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('कृपया आधी लॉगिन करा / Please login first', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if 'user_id' not in session:
        return None
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
    conn.close()
    return user

def get_all_crops():
    conn   = get_db()
    crops  = conn.execute('SELECT * FROM crops ORDER BY name_en').fetchall()
    conn.close()
    return crops


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC PAGES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    user  = get_current_user()
    crops = get_all_crops()
    return render_template('home.html', user=user, crops=crops)

@app.route('/about')
def about():
    return render_template('about.html', user=get_current_user())


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name     = (request.form.get('name') or '').strip()
        phone    = (request.form.get('phone') or '').strip()
        village  = (request.form.get('village') or '').strip()
        district = (request.form.get('district') or '').strip()
        password = (request.form.get('password') or '')

        # Keep only digits in phone
        phone_digits = ''.join(filter(str.isdigit, phone))

        if not name:
            flash('❌ नाव भरणे आवश्यक आहे / Name is required', 'danger')
            return render_template('register.html')
        if len(phone_digits) < 10:
            flash('❌ १० अंकी मोबाइल नंबर भरा / Enter valid 10-digit mobile number', 'danger')
            return render_template('register.html')
        if len(password) < 4:
            flash('❌ पासवर्ड किमान ४ अक्षरांचा असावा / Password must be at least 4 characters', 'danger')
            return render_template('register.html')

        try:
            conn = get_db()
            existing = conn.execute('SELECT id FROM users WHERE phone=?', (phone_digits,)).fetchone()
            if existing:
                conn.close()
                flash('❌ हा मोबाइल नंबर आधीच नोंदणीकृत आहे / Phone already registered. Please login.', 'danger')
                return render_template('register.html')

            conn.execute(
                'INSERT INTO users (name, phone, village, district, password_hash, lang) VALUES (?,?,?,?,?,?)',
                (name, phone_digits, village, district, hash_pw(password), 'mr')
            )
            conn.commit()
            conn.close()
            flash(f'✅ नोंदणी यशस्वी! {name} जी, आता लॉगिन करा / Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'❌ नोंदणी अयशस्वी / Registration failed: {str(e)}', 'danger')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        phone    = ''.join(filter(str.isdigit, (request.form.get('phone') or '')))
        password = (request.form.get('password') or '')

        if not phone:
            flash('❌ मोबाइल नंबर भरा / Enter mobile number', 'danger')
            return render_template('login.html')
        if not password:
            flash('❌ पासवर्ड भरा / Enter password', 'danger')
            return render_template('login.html')

        try:
            conn         = get_db()
            phone_exists = conn.execute('SELECT id FROM users WHERE phone=?', (phone,)).fetchone()
            if not phone_exists:
                conn.close()
                flash('❌ हा नंबर नोंदणीकृत नाही / Phone not registered. Please register first.', 'danger')
                return render_template('login.html')

            user = conn.execute(
                'SELECT * FROM users WHERE phone=? AND password_hash=?',
                (phone, hash_pw(password))
            ).fetchone()
            conn.close()

            if user:
                session.clear()
                session['user_id']   = user['id']
                session['user_name'] = user['name']
                session.permanent    = True
                flash(f'✅ स्वागत आहे {user["name"]}! / Welcome {user["name"]}!', 'success')
                return redirect(url_for('dashboard'))

            flash('❌ चुकीचा पासवर्ड / Wrong password. Try again.', 'danger')
        except Exception as e:
            flash(f'❌ लॉगिन त्रुटी / Login error: {str(e)}', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('लॉगआउट यशस्वी / Logged out successfully', 'info')
    return redirect(url_for('home'))


# ══════════════════════════════════════════════════════════════════════════════
#  FARMER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/dashboard')
@login_required
def dashboard():
    uid  = session['user_id']
    conn = get_db()

    # Summary stats
    total_sales = conn.execute(
        'SELECT COALESCE(SUM(total_amount),0) as t FROM sales WHERE user_id=?',(uid,)
    ).fetchone()['t']
    total_profit = conn.execute(
        'SELECT COALESCE(SUM(profit),0) as t FROM sales WHERE user_id=?',(uid,)
    ).fetchone()['t']
    active_stock = conn.execute(
        'SELECT COUNT(*) as c FROM stock WHERE user_id=? AND status="stored"',(uid,)
    ).fetchone()['c']
    pred_count = conn.execute(
        'SELECT COUNT(*) as c FROM predictions WHERE user_id=?',(uid,)
    ).fetchone()['c']

    # Recent sales (last 5)
    recent_sales = conn.execute('''
        SELECT s.*, c.name_en, c.name_mr FROM sales s
        JOIN crops c ON s.crop_id=c.id
        WHERE s.user_id=? ORDER BY s.created_at DESC LIMIT 5
    ''', (uid,)).fetchall()

    # Monthly profit chart data (last 6 months)
    monthly = conn.execute('''
        SELECT strftime('%Y-%m', sale_date) as mo, SUM(profit) as p
        FROM sales WHERE user_id=?
        GROUP BY mo ORDER BY mo DESC LIMIT 6
    ''', (uid,)).fetchall()

    # Stock summary
    stocks = conn.execute('''
        SELECT st.*, c.name_en, c.name_mr FROM stock st
        JOIN crops c ON st.crop_id=c.id
        WHERE st.user_id=? AND st.status="stored"
        ORDER BY st.created_at DESC LIMIT 5
    ''', (uid,)).fetchall()

    conn.close()

    # Convert SQLite Row objects to plain dicts (required for tojson in template)
    recent_sales = [dict(r) for r in recent_sales]
    monthly      = [{'mo': r['mo'], 'p': float(r['p'] or 0)} for r in monthly]
    stocks       = [dict(r) for r in stocks]

    return render_template('dashboard.html',
        user=get_current_user(),
        total_sales=round(float(total_sales or 0), 2),
        total_profit=round(float(total_profit or 0), 2),
        active_stock=int(active_stock or 0),
        pred_count=int(pred_count or 0),
        recent_sales=recent_sales,
        monthly=monthly,
        stocks=stocks
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/stock')
@login_required
def stock():
    uid    = session['user_id']
    conn   = get_db()
    stocks = conn.execute('''
        SELECT st.*, c.name_en, c.name_mr, c.category FROM stock st
        JOIN crops c ON st.crop_id=c.id
        WHERE st.user_id=? ORDER BY st.created_at DESC
    ''', (uid,)).fetchall()
    crops  = get_all_crops()
    conn.close()
    return render_template('stock.html', user=get_current_user(),
                           stocks=stocks, crops=crops)


@app.route('/stock/add', methods=['POST'])
@login_required
def add_stock():
    uid = session['user_id']
    try:
        def safe_float(val, default=0.0):
            """Convert form value to float safely — handles empty strings."""
            try:
                return float(val) if val and str(val).strip() != '' else default
            except (ValueError, TypeError):
                return default

        crop_id      = int(request.form.get('crop_id', 1))
        quantity_kg  = safe_float(request.form.get('quantity_kg'), 0.0)
        if quantity_kg <= 0:
            flash('प्रमाण (quantity) योग्य भरा / Please enter valid quantity', 'danger')
            return redirect(url_for('stock'))

        conn = get_db()
        conn.execute('''
            INSERT INTO stock
            (user_id, crop_id, quantity_kg, purchase_price, production_cost,
             harvest_date, storage_type, notes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'stored')
        ''', (
            uid,
            crop_id,
            quantity_kg,
            safe_float(request.form.get('purchase_price')),
            safe_float(request.form.get('production_cost')),
            request.form.get('harvest_date', '').strip(),
            request.form.get('storage_type', 'home'),
            request.form.get('notes', '').strip(),
        ))
        conn.commit()
        conn.close()
        flash('✅ माल साठा यशस्वीरित्या जोडला! / Stock added successfully!', 'success')
    except Exception as e:
        flash(f'Error saving stock: {str(e)}', 'danger')
    return redirect(url_for('stock'))


@app.route('/stock/sell/<int:stock_id>', methods=['POST'])
@login_required
def sell_stock(stock_id):
    uid = session['user_id']
    try:
        conn  = get_db()
        st    = conn.execute('SELECT * FROM stock WHERE id=? AND user_id=?',
                             (stock_id, uid)).fetchone()
        if not st:
            flash('Stock not found', 'danger')
            return redirect(url_for('stock'))

        qty        = float(request.form.get('quantity_kg') or 0)
        sale_price = float(request.form.get('sale_price') or 0)
        if qty <= 0 or sale_price <= 0:
            flash('योग्य प्रमाण आणि भाव भरा / Enter valid quantity and price', 'danger')
            return redirect(url_for('stock'))
        total  = qty * sale_price
        cost   = float(st['production_cost'] or 0)
        profit = total - cost * qty

        conn.execute('''
            INSERT INTO sales
            (user_id,stock_id,crop_id,quantity_kg,sale_price,total_amount,
             buyer,sale_date,profit,notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        ''', (uid, stock_id, st['crop_id'], qty, sale_price, total,
              request.form.get('buyer',''),
              request.form.get('sale_date', datetime.date.today().isoformat()),
              round(profit, 2), request.form.get('notes','')))

        # Update stock quantity
        new_qty = st['quantity_kg'] - qty
        if new_qty <= 0:
            conn.execute('UPDATE stock SET status="sold" WHERE id=?', (stock_id,))
        else:
            conn.execute('UPDATE stock SET quantity_kg=? WHERE id=?', (new_qty, stock_id))

        conn.commit()
        conn.close()
        flash(f'विक्री नोंदवली! नफा: ₹{profit:.0f} / Sale recorded! Profit: ₹{profit:.0f}', 'success')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('stock'))


# ══════════════════════════════════════════════════════════════════════════════
#  SALES HISTORY
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/sales')
@login_required
def sales():
    uid   = session['user_id']
    conn  = get_db()
    rows  = conn.execute('''
        SELECT s.*, c.name_en, c.name_mr FROM sales s
        JOIN crops c ON s.crop_id=c.id
        WHERE s.user_id=? ORDER BY s.created_at DESC
    ''', (uid,)).fetchall()

    total_profit = conn.execute(
        'SELECT COALESCE(SUM(profit),0) as t FROM sales WHERE user_id=?',(uid,)
    ).fetchone()['t']
    conn.close()
    return render_template('sales.html', user=get_current_user(),
                           sales=rows, total_profit=total_profit)


# ══════════════════════════════════════════════════════════════════════════════
#  AI ADVISOR PAGE
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/advisor')
def advisor():
    crops = get_all_crops()
    return render_template('advisor.html', user=get_current_user(), crops=crops)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        d = request.get_json() or request.form.to_dict()

        crop_name = str(d.get('crop_name', 'onion')).lower().strip()
        qty       = float(d.get('quantity', 100) or 100)
        buy_price = float(d.get('current_price', 10) or 10)
        prod_cost = float(d.get('production_cost', 0) or 0)
        transport = float(d.get('transport_cost', 2) or 2)
        storage   = float(d.get('storage_cost', 2) or 2)
        temp      = str(d.get('temperature', 'medium') or 'medium')

        # Always resolve crop_id from DB by name — prevents FOREIGN KEY error
        conn     = get_db()
        crop_row = conn.execute('SELECT id FROM crops WHERE name_en=?', (crop_name,)).fetchone()
        if not crop_row:
            crop_row = conn.execute('SELECT id FROM crops WHERE name_en LIKE ?', (f'%{crop_name}%',)).fetchone()
        crop_id = crop_row['id'] if crop_row else 1
        conn.close()

        user_prices = []
        for k in ['past_price_1', 'past_price_2', 'past_price_3']:
            v = d.get(k)
            if v:
                try: user_prices.append(float(v))
                except: pass

        result = full_analysis(
            crop_id, crop_name, qty, buy_price, prod_cost,
            transport, storage, temp,
            user_prices if user_prices else None
        )

        # Save to DB — wrapped so it never blocks the result
        if 'user_id' in session:
            try:
                conn = get_db()
                conn.execute('''
                    INSERT INTO predictions
                    (user_id,crop_id,quantity,current_price,best_strategy,
                     expected_profit,risk_level,result_json)
                    VALUES (?,?,?,?,?,?,?,?)
                ''', (session['user_id'], crop_id, qty, buy_price,
                      result['best_strategy'], result['expected_profit'],
                      result['risk'], json.dumps(result)))
                conn.commit()
                conn.close()
            except Exception:
                pass  # logging failure must never block the AI result

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/crops')
def api_crops():
    crops = get_all_crops()
    return jsonify([dict(c) for c in crops])


# ══════════════════════════════════════════════════════════════════════════════
#  PROFILE
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    uid = session['user_id']
    if request.method == 'POST':
        conn = get_db()
        conn.execute('''
            UPDATE users SET name=?,village=?,district=?,lang=?
            WHERE id=?
        ''', (request.form['name'], request.form.get('village',''),
              request.form.get('district',''),
              request.form.get('lang','mr'), uid))
        conn.commit()
        conn.close()
        session['user_name'] = request.form['name']
        flash('प्रोफाइल अपडेट झाली / Profile updated', 'success')
    return render_template('profile.html', user=get_current_user())


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    seed_price_history()
    from ai_engine import pretrain_all_models
    pretrain_all_models()
    print("=" * 60)
    print("  🌿 KrishiAdvisor AI V2")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

