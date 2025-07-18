from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
import ast
import random
from collections import defaultdict


# === FLASK APP ===
app = Flask(__name__)

# === LOAD & PARSE DATASET ===
df = pd.read_csv('reseptrainingg.csv')
print("CSV Loaded:", df.shape)

# Fungsi membersihkan Ingredients dari CSV
def clean_ingredient_text(s):
    try:
        if isinstance(s, str):
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
    return name.split()[0]

# Parsing kuantitas bahan
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

def ingredient_to_vector(parsed):
    return [parsed.get(i, 0) for i in ingredient_columns]

X_ingredients = np.array([ingredient_to_vector(p) for p in df['Parsed']])

# Latih model KNN
knn = None
if len(ingredient_columns) > 0:
    knn = NearestNeighbors(n_neighbors=len(df), metric='euclidean')
    knn.fit(X_ingredients)
    print("KNN model trained successfully")

# === FUNGSI REKOMENDASI DENGAN SISTEM PRIORITAS BARU ===
def ingredients_match(user_ing, recipe_ing):
    """Cek apakah dua bahan cocok (exact match atau substring)"""
    return (user_ing == recipe_ing or 
            user_ing in recipe_ing or 
            recipe_ing in user_ing)

def calculate_gramasi_accuracy(user_input, recipe_parsed, matching_ingredients):
    """Hitung akurasi gramasi berdasarkan bahan yang cocok"""
    if not matching_ingredients:
        return 0.0
    
    accuracies = []
    for user_ing, recipe_ing in matching_ingredients:
        user_qty = user_input[user_ing]
        recipe_qty = recipe_parsed[recipe_ing]
        
        if recipe_qty > 0:
            ratio = user_qty / recipe_qty
            # Semakin dekat dengan 1, semakin akurat
            accuracy = 1.0 / (1.0 + abs(ratio - 1.0))
            accuracies.append(accuracy)
    
    return sum(accuracies) / len(accuracies) if accuracies else 0.0

def calculate_scaling_factor(user_input, recipe_parsed):
    scaling_factors = []
    for user_ing, user_qty in user_input.items():
        for recipe_ing, recipe_qty in recipe_parsed.items():
            if ingredients_match(user_ing, recipe_ing):
                if recipe_qty > 0:
                    factor = user_qty / recipe_qty
                    scaling_factors.append(factor)
    if scaling_factors:
        return round(sum(scaling_factors) / len(scaling_factors), 2)
    return 1.0

def scale_recipe_ingredients(recipe_parsed, scale_factor):
    scaled_ingredients = {}
    for ingredient, qty in recipe_parsed.items():
        scaled_ingredients[ingredient] = round(qty * scale_factor, 2)
    return scaled_ingredients

def recommend_recipe(user_input, max_recommendations=20):
    """
    Sistem rekomendasi dengan prioritas ketat:
    1. Bahan pertama HARUS ada dalam setiap rekomendasi
    2. Prioritas 1: Semua bahan cocok (diurutkan berdasarkan gramasi)
    3. Prioritas 2: Bahan pertama + sebagian bahan lain (diurutkan berdasarkan gramasi)
    4. Prioritas 3: Hanya bahan pertama (diurutkan berdasarkan gramasi)
    """
    if knn is None or len(ingredient_columns) == 0 or not user_input:
        return []
    
    # Ambil bahan pertama sebagai bahan WAJIB
    user_ingredients_order = list(user_input.keys())
    primary_ingredient = user_ingredients_order[0]
    
    print(f"DEBUG: Primary ingredient (WAJIB): {primary_ingredient}")
    print(f"DEBUG: User input order: {user_ingredients_order}")
    
    # Cari semua resep yang mengandung bahan pertama
    primary_recipes = []
    
    user_vector = np.array([user_input.get(i, 0) for i in ingredient_columns]).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector)
    
    for idx in indices[0]:
        resep = df.iloc[idx]
        recipe_parsed = resep['Parsed']
        
        # CEK WAJIB: Bahan pertama harus ada
        primary_found = False
        primary_match_pair = None
        
        for recipe_ing in recipe_parsed.keys():
            if ingredients_match(primary_ingredient, recipe_ing):
                primary_found = True
                primary_match_pair = (primary_ingredient, recipe_ing)
                break
        
        # Skip jika bahan pertama tidak ada
        if not primary_found:
            continue
        
        # Hitung matching ingredients dengan urutan prioritas
        matching_ingredients = []
        matched_user_ingredients = set()
        
        # Prioritas matching berdasarkan urutan input user
        for user_ing in user_ingredients_order:
            for recipe_ing in recipe_parsed.keys():
                if ingredients_match(user_ing, recipe_ing):
                    if user_ing not in matched_user_ingredients:
                        matching_ingredients.append((user_ing, recipe_ing))
                        matched_user_ingredients.add(user_ing)
                        break
        
        # Hitung porsi maksimal yang bisa dibuat
        available_portions = []
        for user_ing, user_qty in user_input.items():
            for recipe_ing, recipe_qty in recipe_parsed.items():
                if ingredients_match(user_ing, recipe_ing) and recipe_qty > 0:
                    portion = user_qty / recipe_qty
                    available_portions.append(portion)
                    break
        
        max_porsi = round(min(available_portions) if available_portions else 0, 2)
        if max_porsi <= 0:
            continue
        
        # Hitung akurasi gramasi
        gramasi_accuracy = calculate_gramasi_accuracy(user_input, recipe_parsed, matching_ingredients)
        
        # Hitung scaling factor
        scale_factor = calculate_scaling_factor(user_input, recipe_parsed)
        scaled_ingredients = scale_recipe_ingredients(recipe_parsed, scale_factor)
        
        # Tentukan prioritas berdasarkan bahan yang cocok
        ingredient_match_count = len(matched_user_ingredients)
        total_user_ingredients = len(user_input)
        
        if ingredient_match_count == total_user_ingredients:
            priority_level = 1  # Semua bahan cocok
        elif ingredient_match_count > 1:  # Bahan pertama + minimal 1 bahan lain
            priority_level = 2  # Bahan pertama + sebagian
        else:  # Hanya bahan pertama
            priority_level = 3  # Hanya bahan pertama
        
        recipe_data = {
            'title': resep['Title'],
            'original_ingredients': recipe_parsed,
            'scaled_ingredients': scaled_ingredients,
            'steps': resep.get('Steps', 'No steps available'),
            'porsi': max_porsi,
            'scale_factor': scale_factor,
            'distance': distances[0][len(primary_recipes)],
            'ingredient_match_count': ingredient_match_count,
            'ingredient_match_ratio': ingredient_match_count / total_user_ingredients,
            'matching_ingredients': matching_ingredients,
            'gramasi_accuracy': gramasi_accuracy,
            'priority_level': priority_level,
            'primary_ingredient_match': primary_match_pair
        }
        
        primary_recipes.append(recipe_data)
    
    # Sorting berdasarkan prioritas dan gramasi
    def sort_key(recipe):
        return (
            recipe['priority_level'],           # Prioritas 1-3
            -recipe['gramasi_accuracy'],        # Gramasi terbaik dulu
            -recipe['ingredient_match_count'],  # Lebih banyak bahan cocok
            -recipe['porsi'],                   # Porsi lebih besar
            recipe['distance']                  # Jarak terdekat
        )
    
    primary_recipes.sort(key=sort_key)
    
    # Batasi hasil maksimal
    final_recommendations = primary_recipes[:max_recommendations]
    
    print(f"DEBUG: Found {len(primary_recipes)} recipes with primary ingredient")
    print(f"DEBUG: Returning {len(final_recommendations)} recommendations")
    
    # Debug output untuk 5 teratas
    for i, rec in enumerate(final_recommendations[:5]):
        print(f"DEBUG: {i+1}. {rec['title']} - Priority: {rec['priority_level']}, " +
              f"Gramasi: {rec['gramasi_accuracy']:.3f}, Matches: {rec['ingredient_match_count']}")
    
    return final_recommendations

def display_recommendation_debug(user_input, recommendations, top_n=5):
    """Fungsi untuk debugging rekomendasi"""
    print(f"\n=== DEBUGGING REKOMENDASI DENGAN PRIORITAS KETAT ===")
    print(f"Input User: {user_input}")
    print(f"Urutan prioritas bahan: {list(user_input.keys())}")
    print(f"Bahan WAJIB (pertama): {list(user_input.keys())[0]}")
    print(f"Total rekomendasi: {len(recommendations)}")
    
    # Hitung distribusi prioritas
    priority_counts = {}
    for rec in recommendations:
        priority = rec['priority_level']
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print(f"\nDistribusi Prioritas:")
    priority_names = {1: "Semua bahan cocok", 2: "Bahan pertama + sebagian", 3: "Hanya bahan pertama"}
    for priority in sorted(priority_counts.keys()):
        print(f"  Prioritas {priority} ({priority_names[priority]}): {priority_counts[priority]} resep")
    
    print(f"\nTop {top_n} Rekomendasi:")
    current_priority = None
    
    for i, rec in enumerate(recommendations[:top_n]):
        if rec['priority_level'] != current_priority:
            current_priority = rec['priority_level']
            print(f"\n--- PRIORITAS {current_priority}: {priority_names[current_priority]} ---")
        
        print(f"\n{i+1}. {rec['title']}")
        print(f"   Bahan cocok: {rec['ingredient_match_count']}/{len(user_input)} ({rec['ingredient_match_ratio']:.2%})")
        print(f"   Matching ingredients: {rec['matching_ingredients']}")
        print(f"   Gramasi accuracy: {rec['gramasi_accuracy']:.3f}")
        print(f"   Porsi maksimal: {rec['porsi']:.2f}")
        print(f"   Scale factor: {rec['scale_factor']:.2f}")
        print(f"   Distance: {rec['distance']:.2f}")
        print(f"   Bahan resep: {list(rec['original_ingredients'].keys())}")
        
        # Tampilkan perbandingan kuantitas untuk bahan yang cocok
        print(f"   Perbandingan kuantitas:")
        for user_ing, recipe_ing in rec['matching_ingredients']:
            user_qty = user_input[user_ing]
            recipe_qty = rec['original_ingredients'][recipe_ing]
            ratio = user_qty / recipe_qty
            accuracy = 1.0 / (1.0 + abs(ratio - 1.0))
            print(f"     {user_ing} ({user_qty}) vs {recipe_ing} ({recipe_qty}) = {ratio:.2f}x (accuracy: {accuracy:.3f})")

# === TESTING FUNCTIONS ===
class RecipeModelTester:
    def __init__(self):
        self.test_results = {}
    
    def create_test_scenarios(self, n_scenarios=100):
        """Buat skenario testing dengan prioritas bahan"""
        test_scenarios = []
        
        for i in range(n_scenarios):
            # Pilih resep random sebagai target
            recipe_idx = random.randint(0, len(df) - 1)
            target_recipe = df.iloc[recipe_idx]
            recipe_ingredients = target_recipe['Parsed']
            
            if not recipe_ingredients or len(recipe_ingredients) < 2:
                continue
            
            # Pilih 2-4 bahan dengan urutan prioritas
            available_ingredients = list(recipe_ingredients.keys())
            num_ingredients = min(random.randint(2, 4), len(available_ingredients))
            selected_ingredients = random.sample(available_ingredients, num_ingredients)
            
            # Buat user stock dengan urutan prioritas
            user_stock = {}
            for ingredient in selected_ingredients:
                original_qty = recipe_ingredients[ingredient]
                user_stock[ingredient] = original_qty * random.uniform(0.8, 1.2)
            
            test_scenarios.append({
                'user_stock': user_stock,
                'target_recipe': target_recipe['Title'],
                'target_index': recipe_idx,
                'target_ingredients': recipe_ingredients,
                'primary_ingredient': selected_ingredients[0]
            })
        
        return test_scenarios
    
    def calculate_ingredient_overlap(self, user_stock, recipe_parsed):
        """Hitung overlap bahan (Jaccard similarity)"""
        if not recipe_parsed or not user_stock:
            return 0.0
        
        user_ingredients = set(user_stock.keys())
        recipe_ingredients = set(recipe_parsed.keys())
        
        intersection = len(user_ingredients.intersection(recipe_ingredients))
        union = len(user_ingredients.union(recipe_ingredients))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_recommendations(self, test_scenarios, top_k=5):
        """Evaluasi sistem rekomendasi dengan prioritas ketat"""
        results = {
            'hit_rate': [],
            'mrr': [],
            'precision_at_k': [],
            'primary_ingredient_coverage': [],  # Apakah bahan pertama selalu ada
            'priority_distribution': [],
            'gramasi_accuracy': []
        }
        
        for scenario in test_scenarios:
            user_stock = scenario['user_stock']
            target_recipe = scenario['target_recipe']
            target_ingredients = scenario['target_ingredients']
            primary_ingredient = scenario['primary_ingredient']
            
            recommendations = recommend_recipe(user_stock)
            
            if not recommendations:
                results['hit_rate'].append(0)
                results['mrr'].append(0)
                results['precision_at_k'].append(0)
                results['primary_ingredient_coverage'].append(0)
                results['priority_distribution'].append(0)
                results['gramasi_accuracy'].append(0)
                continue
            
            # 1. Hit Rate
            top_k_titles = [rec['title'] for rec in recommendations[:top_k]]
            hit = 1 if target_recipe in top_k_titles else 0
            results['hit_rate'].append(hit)
            
            # 2. Mean Reciprocal Rank
            mrr = 0
            for i, title in enumerate(top_k_titles):
                if title == target_recipe:
                    mrr = 1 / (i + 1)
                    break
            results['mrr'].append(mrr)
            
            # 3. Precision@k
            overlaps = []
            for rec in recommendations[:top_k]:
                overlap = self.calculate_ingredient_overlap(user_stock, rec['original_ingredients'])
                overlaps.append(overlap)
            
            precision = sum(overlaps) / len(overlaps) if overlaps else 0
            results['precision_at_k'].append(precision)
            
            # 4. Primary Ingredient Coverage
            primary_coverage = 1  # Dengan sistem baru, ini selalu 1
            for rec in recommendations[:top_k]:
                has_primary = False
                for user_ing, recipe_ing in rec['matching_ingredients']:
                    if user_ing == primary_ingredient:
                        has_primary = True
                        break
                if not has_primary:
                    primary_coverage = 0
                    break
            results['primary_ingredient_coverage'].append(primary_coverage)
            
            # 5. Priority Distribution
            priority_scores = []
            for rec in recommendations[:top_k]:
                priority_scores.append(4 - rec['priority_level'])  # Inversi agar skor tinggi = prioritas tinggi
            
            avg_priority = sum(priority_scores) / len(priority_scores) if priority_scores else 0
            results['priority_distribution'].append(avg_priority)
            
            # 6. Gramasi Accuracy
            gramasi_scores = []
            for rec in recommendations[:top_k]:
                gramasi_scores.append(rec['gramasi_accuracy'])
            
            avg_gramasi = sum(gramasi_scores) / len(gramasi_scores) if gramasi_scores else 0
            results['gramasi_accuracy'].append(avg_gramasi)
        
        return results
    
    def run_comprehensive_test(self, n_scenarios=100):
        """Jalankan test komprehensif dengan prioritas ketat"""
        print("=" * 60)
        print("TESTING RECIPE RECOMMENDER WITH STRICT PRIORITY SYSTEM")
        print("=" * 60)
        
        print(f"Creating {n_scenarios} test scenarios...")
        test_scenarios = self.create_test_scenarios(n_scenarios)
        
        if not test_scenarios:
            print("ERROR: No valid test scenarios created!")
            return
        
        print(f"Evaluating {len(test_scenarios)} scenarios...")
        results = self.evaluate_recommendations(test_scenarios)
        
        self.test_results = {
            'overall': results,
            'n_scenarios': len(test_scenarios)
        }
        
        self.print_results()
        return results
    
    def print_results(self):
        """Print hasil testing"""
        results = self.test_results['overall']
        
        print("\n" + "=" * 60)
        print("STRICT PRIORITY SYSTEM PERFORMANCE METRICS")
        print("=" * 60)
        
        print(f"Test Scenarios: {self.test_results['n_scenarios']}")
        print(f"Dataset Size: {len(df)} recipes")
        print(f"Total Ingredients: {len(ingredient_columns)}")
        
        print(f"\nCORE METRICS:")
        print(f"Hit Rate@5: {np.mean(results['hit_rate']):.3f} ± {np.std(results['hit_rate']):.3f}")
        print(f"  → {np.mean(results['hit_rate'])*100:.1f}% of target recipes found in top-5")
        
        print(f"\nMean Reciprocal Rank: {np.mean(results['mrr']):.3f} ± {np.std(results['mrr']):.3f}")
        print(f"  → Average position of target recipe: {1/np.mean(results['mrr']):.1f}" if np.mean(results['mrr']) > 0 else "  → Target recipes not found")
        
        print(f"\nPrecision@5: {np.mean(results['precision_at_k']):.3f} ± {np.std(results['precision_at_k']):.3f}")
        print(f"  → Average ingredient overlap in top-5: {np.mean(results['precision_at_k'])*100:.1f}%")
        
        print(f"\nPrimary Ingredient Coverage: {np.mean(results['primary_ingredient_coverage']):.3f}")
        print(f"  → {np.mean(results['primary_ingredient_coverage'])*100:.1f}% of recommendations contain primary ingredient")
        
        print(f"\nPriority Distribution Score: {np.mean(results['priority_distribution']):.3f}")
        print(f"  → Average priority quality (3=best, 0=worst)")
        
        print(f"\nGramasi Accuracy: {np.mean(results['gramasi_accuracy']):.3f} ± {np.std(results['gramasi_accuracy']):.3f}")
        print(f"  → Average quantity matching accuracy: {np.mean(results['gramasi_accuracy'])*100:.1f}%")
        
        # Interpretasi hasil
        print(f"\nSYSTEM EVALUATION:")
        primary_coverage = np.mean(results['primary_ingredient_coverage'])
        hit_rate = np.mean(results['hit_rate'])
        gramasi_accuracy = np.mean(results['gramasi_accuracy'])
        
        if primary_coverage >= 0.95:
            print("✅ EXCELLENT: Primary ingredient constraint strictly enforced")
        elif primary_coverage >= 0.85:
            print("✅ GOOD: Primary ingredient mostly preserved")
        else:
            print("❌ POOR: Primary ingredient constraint violated")
        
        if hit_rate >= 0.6:
            print("✅ EXCELLENT: Very good at finding target recipes")
        elif hit_rate >= 0.4:
            print("✅ GOOD: Good performance on target finding")
        else:
            print("⚠️  FAIR: Moderate performance on target finding")
        
        if gramasi_accuracy >= 0.7:
            print("✅ EXCELLENT: High quantity matching accuracy")
        elif gramasi_accuracy >= 0.5:
            print("✅ GOOD: Moderate quantity matching")
        else:
            print("❌ POOR: Low quantity matching accuracy")

# === FLASK ROUTES ===
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
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
            # Debug output
            display_recommendation_debug(user_stock, recommendations)
    
    return render_template('coba.html', recommendations=recommendations)

@app.route('/test')
def test_model():
    """Route untuk testing model"""
    tester = RecipeModelTester()
    results = tester.run_comprehensive_test(n_scenarios=100)
    
    return f"<pre>{tester.test_results}</pre>"

# === RUN SERVER ===
if __name__ == '__main__':
    print("Training model...")
    
    # Jalankan test otomatis saat startup
    print("\nRunning automatic strict priority system test...")
    tester = RecipeModelTester()
    tester.run_comprehensive_test(n_scenarios=50)
    
    print("\nStarting Flask server...")
    print("Visit /test for detailed testing results")
    app.run(debug=True)