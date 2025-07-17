from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
import ast

# === FLASK APP ===
app = Flask(__name__)

# === LOAD & PARSE DATASET ===
df = pd.read_csv('reseptrainingg.csv')
print("CSV Loaded:", df.shape)

# Fungsi membersihkan Ingredients dari CSV
def clean_ingredient_text(s):
    try:
        if isinstance(s, str):
            # Coba parsing sebagai Python literal dulu
            try:
                return ast.literal_eval(s)
            except:
                s = s.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1]
                ingredients = []
                for item in s.split(','):
                    item = item.strip().strip('"\'')
                    if item:
                        ingredients.append(item)
                return ingredients
        return []
    except Exception as e:
        print(f"Parsing error: {e}")
        return []

df['Ingredients'] = df['Ingredients'].apply(clean_ingredient_text)

def normalize_ingredient_name(name):
    """Normalisasi nama bahan agar lebih cocok saat matching"""
    name = name.lower().strip()
    mapping = {
        'fillet': 'ayam',
        'daging ayam': 'ayam',
        'ayam fillet': 'ayam',
        'bawang putih': 'bawang',
        'bawang merah': 'bawang',
        'cabe': 'cabai',
        'cabai rawit': 'cabai',
        'santan': 'santan',
        'tepung terigu': 'tepung',
        'tepung sagu': 'tepung',
        'gula pasir': 'gula',
        'gula merah': 'gula',
        'minyak goreng': 'minyak',
        'air putih': 'air'
    }

    for key, value in mapping.items():
        if key in name:
            return value
    return name.split()[0]  # fallback ke kata pertama

# Parsing kuantitas bahan
# === Parsing bahan ===
def parse_ingredients(ingredient_list):
    konversi = {
        'kg': 1000,
        'gram': 1,
        'gr': 1,
        'g': 1,
        'sendok teh': 5,
        'sdt': 5,
        'sdm': 15,
        'butir': 50,
        'biji': 50,
        'siung': 3,
        'lembar': 10,
        'potong': 100,
        'buah': 100,
        'cup': 240,
        'ml': 1,
        'liter': 1000,
        'l': 1000,
        'ekor': 1200,
        'sachet': 10,
        'bungkus': 30,
    }

    ingredient_dict = {}
    if not isinstance(ingredient_list, list):
        return ingredient_dict

    for item in ingredient_list:
        item = item.strip().lower()

        # Support pecahan seperti "1/2 kg"
        pattern = r"([0-9]+(?:/[0-9]+)?(?:\.[0-9]+)?)?\s*(kg|gram|gr|g|sendok teh|sdt|sdm|butir|biji|lembar|potong|siung|buah|cup|ml|liter|l|ekor|sachet|bungkus)?\s*(.*)"
        match = re.match(pattern, item)
        if match:
            qty_str, unit, name = match.groups()
            qty = 1.0
            if qty_str:
                if '/' in qty_str:
                    try:
                        num, denom = qty_str.split('/')
                        qty = float(num) / float(denom)
                    except:
                        qty = 1.0
                else:
                    try:
                        qty = float(qty_str.replace(',', '.'))
                    except:
                        qty = 1.0

            multiplier = konversi.get(unit or '', 1)
            total_qty = qty * multiplier
            main_name = name.split()[0] if name else 'unknown'
            ingredient_dict[main_name] = total_qty
        else:
            words = item.split()
            if words:
                main_name = words[0]
                ingredient_dict[main_name] = 100
    return ingredient_dict




df['Parsed'] = df['Ingredients'].apply(parse_ingredients)

# Ambil semua nama bahan sebagai kolom vektor
all_ingredients = set()
for parsed in df['Parsed']:
    all_ingredients.update(parsed.keys())

ingredient_columns = list(all_ingredients)
print(f"Total unique ingredients: {len(ingredient_columns)}")
if len(ingredient_columns) > 0:
    print(f"Sample ingredients: {ingredient_columns[:10]}")

def ingredient_to_vector(parsed):
    return [parsed.get(i, 0) for i in ingredient_columns]

X_ingredients = np.array([ingredient_to_vector(p) for p in df['Parsed']])

# Latih model KNN
knn = None
if len(ingredient_columns) > 0:
    knn = NearestNeighbors(n_neighbors=len(df), metric='euclidean')  # <= Semua resep
    knn.fit(X_ingredients)
    print("KNN model trained successfully")
else:
    print("ERROR: No ingredients found to train model")

# Hitung porsi maksimal
def calculate_portions(user_stock, recipe_stock):
    porsis = []
    for bahan, qty in recipe_stock.items():
        if bahan in user_stock and qty > 0:
            porsis.append(user_stock[bahan] / qty)
        else:
            porsis.append(0)
    return round(min(porsis) if porsis else 0, 2)

# Fungsi untuk menentukan scaling factor berdasarkan bahan utama
def calculate_scaling_factor(user_input, recipe_parsed):
    """Menghitung faktor scaling berdasarkan bahan utama yang user miliki"""
    scaling_factors = []
    
    for user_ing, user_qty in user_input.items():
        for recipe_ing, recipe_qty in recipe_parsed.items():
            # Cek apakah bahan matching (bisa exact match atau substring)
            if user_ing == recipe_ing or user_ing in recipe_ing or recipe_ing in user_ing:
                if recipe_qty > 0:
                    factor = user_qty / recipe_qty
                    scaling_factors.append(factor)
    
    # Ambil scaling factor yang paling masuk akal (median atau rata-rata)
    if scaling_factors:
        return round(sum(scaling_factors) / len(scaling_factors), 2)
    return 1.0

# Fungsi untuk scaling seluruh resep
def scale_recipe_ingredients(recipe_parsed, scale_factor):
    """Mengalikan semua bahan dalam resep dengan scale_factor"""
    scaled_ingredients = {}
    for ingredient, qty in recipe_parsed.items():
        scaled_ingredients[ingredient] = round(qty * scale_factor, 2)
    return scaled_ingredients

# Rekomendasi utama dengan prioritas scaling
def recommend_recipe(user_input, max_difference=50):
    if knn is None or len(ingredient_columns) == 0:
        return []
    
    exact_matches = []  # Resep yang langsung cocok
    scaled_matches = []  # Resep yang perlu di-scale
    
    # Buat vektor input pengguna
    user_vector = np.array([user_input.get(i, 0) for i in ingredient_columns]).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector)

    for idx in indices[0]:
        resep = df.iloc[idx]
        recipe_parsed = resep['Parsed']
        
        # Cek apakah resep memiliki bahan yang user miliki
        matching_ingredients = []
        for user_ing in user_input.keys():
            for recipe_ing in recipe_parsed.keys():
                if user_ing == recipe_ing or user_ing in recipe_ing or recipe_ing in user_ing:
                    matching_ingredients.append((user_ing, recipe_ing))
        
        if not matching_ingredients:
            continue  # Skip resep yang tidak memiliki bahan yang cocok
        
        # Hitung scaling factor
        scale_factor = calculate_scaling_factor(user_input, recipe_parsed)
        
        # Hitung porsi yang bisa dibuat dengan bahan user
        available_portions = []
        for user_ing, user_qty in user_input.items():
            for recipe_ing, recipe_qty in recipe_parsed.items():
                if user_ing == recipe_ing or user_ing in recipe_ing or recipe_ing in user_ing:
                    if recipe_qty > 0:
                        portion = user_qty / recipe_qty
                        available_portions.append(portion)
                    break

        max_porsi = round(min(available_portions) if available_portions else 0, 2)
        if max_porsi <= 0:
            continue

        # Scaled ingredients untuk ditampilkan
        scaled_ingredients = scale_recipe_ingredients(recipe_parsed, scale_factor)
        
        recipe_data = {
            'title': resep['Title'],
            'original_ingredients': recipe_parsed,
            'scaled_ingredients': scaled_ingredients,
            'steps': resep.get('Steps', 'No steps available'),
            'porsi': max_porsi,
            'scale_factor': scale_factor,
            'distance': distances[0][len(exact_matches + scaled_matches)]
        }
        
        # Kategorikan berdasarkan scaling factor
        if 0.9 <= scale_factor <= 1.1:  # Hampir 1x (toleransi 10%)
            exact_matches.append(recipe_data)
        else:
            scaled_matches.append(recipe_data)
    
    # Sort exact matches by porsi (descending), scaled matches by scale factor (ascending, prioritas ke yang lebih kecil)
    exact_matches.sort(key=lambda x: x['porsi'], reverse=True)
    scaled_matches.sort(key=lambda x: (abs(x['scale_factor'] - 1), -x['porsi']))
    
    # Gabungkan hasil: exact matches dulu, lalu scaled matches
    return exact_matches + scaled_matches


# Rekomendasi dengan bahan tambahan
def recommend_with_additional_ingredients(user_input):
    if knn is None or len(ingredient_columns) == 0:
        return []
    
    results = []
    user_vector = np.array([user_input.get(i, 0) for i in ingredient_columns]).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector)

    for idx in indices[0]:
        resep = df.iloc[idx]
        recipe_parsed = resep['Parsed']
        
        missing = [b for b in recipe_parsed if b not in user_input]
        if missing:
            results.append({
                'title': resep['Title'],
                'missing': missing,
                'steps': resep.get('Steps', 'No steps available')
            })
    
    return results

# === ROUTE FLASK ===
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    additional_recommendations = []
    if request.method == 'POST':
        names = request.form.getlist('ingredient_name[]')
        qtys = request.form.getlist('ingredient_qty[]')
        
        user_stock = {}
        for name, qty in zip(names, qtys):
            if name.strip() and qty.strip():
                try:
                    normalized_name = normalize_ingredient_name(name)
                    user_stock[normalized_name] = float(qty)
                except ValueError:
                    continue
        
        if user_stock:
            recommendations = recommend_recipe(user_stock)
            show_additional = request.form.get('show_additional_recipes') == 'on'
            if show_additional:
                additional_recommendations = recommend_with_additional_ingredients(user_stock)
    
    return render_template('index.html',
                           recommendations=recommendations,
                           additional_recommendations=additional_recommendations)

# === RUN SERVER ===
if __name__ == '__main__':
    app.run(debug=True)