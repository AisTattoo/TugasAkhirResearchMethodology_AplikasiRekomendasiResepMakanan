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

# === FUNGSI REKOMENDASI (SAMA SEPERTI SEBELUMNYA) ===
def calculate_scaling_factor(user_input, recipe_parsed):
    scaling_factors = []
    for user_ing, user_qty in user_input.items():
        for recipe_ing, recipe_qty in recipe_parsed.items():
            if user_ing == recipe_ing or user_ing in recipe_ing or recipe_ing in user_ing:
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

def recommend_recipe(user_input, max_difference=50):
    if knn is None or len(ingredient_columns) == 0:
        return []
    
    exact_matches = []
    scaled_matches = []
    
    user_vector = np.array([user_input.get(i, 0) for i in ingredient_columns]).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector)

    for idx in indices[0]:
        resep = df.iloc[idx]
        recipe_parsed = resep['Parsed']
        
        matching_ingredients = []
        for user_ing in user_input.keys():
            for recipe_ing in recipe_parsed.keys():
                if user_ing == recipe_ing or user_ing in recipe_ing or recipe_ing in user_ing:
                    matching_ingredients.append((user_ing, recipe_ing))
        
        if not matching_ingredients:
            continue
        
        scale_factor = calculate_scaling_factor(user_input, recipe_parsed)
        
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
        
        if 0.9 <= scale_factor <= 1.1:
            exact_matches.append(recipe_data)
        else:
            scaled_matches.append(recipe_data)
    
    exact_matches.sort(key=lambda x: x['porsi'], reverse=True)
    scaled_matches.sort(key=lambda x: (abs(x['scale_factor'] - 1), -x['porsi']))
    
    return exact_matches + scaled_matches

# === TESTING FUNCTIONS ===
class RecipeModelTester:
    def __init__(self):
        self.test_results = {}
    
    def create_test_scenarios(self, n_scenarios=100):
        """Buat skenario testing"""
        test_scenarios = []
        
        for i in range(n_scenarios):
            # Pilih resep random sebagai target
            recipe_idx = random.randint(0, len(df) - 1)
            target_recipe = df.iloc[recipe_idx]
            recipe_ingredients = target_recipe['Parsed']
            
            if not recipe_ingredients:
                continue
            
            # User punya 60-80% bahan dari resep
            num_ingredients = max(1, int(len(recipe_ingredients) * random.uniform(0.6, 0.8)))
            available_ingredients = random.sample(list(recipe_ingredients.keys()), num_ingredients)
            
            # Quantity 80-120% dari yang dibutuhkan
            user_stock = {}
            for ingredient in available_ingredients:
                original_qty = recipe_ingredients[ingredient]
                user_stock[ingredient] = original_qty * random.uniform(0.8, 1.2)
            
            test_scenarios.append({
                'user_stock': user_stock,
                'target_recipe': target_recipe['Title'],
                'target_index': recipe_idx,
                'target_ingredients': recipe_ingredients
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
        """
        Evaluasi menggunakan berbagai metrik:
        1. Hit Rate - berapa % target recipe masuk top-k
        2. MRR - posisi rata-rata target recipe dalam ranking
        3. Precision@k - rata-rata relevansi top-k recommendations
        4. Recall@k - seberapa baik sistem menemukan resep yang relevan
        5. Coverage - berapa % total resep yang pernah direkomendasikan
        """
        results = {
            'hit_rate': [],
            'mrr': [],  # Mean Reciprocal Rank
            'precision_at_k': [],
            'recall_at_k': [],
            'ingredient_overlap': [],
            'portion_accuracy': []
        }
        
        recommended_recipes = set()
        
        for scenario in test_scenarios:
            user_stock = scenario['user_stock']
            target_recipe = scenario['target_recipe']
            target_ingredients = scenario['target_ingredients']
            
            recommendations = recommend_recipe(user_stock)
            
            if not recommendations:
                # Jika tidak ada rekomendasi
                results['hit_rate'].append(0)
                results['mrr'].append(0)
                results['precision_at_k'].append(0)
                results['recall_at_k'].append(0)
                results['ingredient_overlap'].append(0)
                results['portion_accuracy'].append(0)
                continue
            
            # Track coverage
            for rec in recommendations[:top_k]:
                recommended_recipes.add(rec['title'])
            
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
            
            # 3. Precision@k (berdasarkan ingredient overlap)
            overlaps = []
            for rec in recommendations[:top_k]:
                overlap = self.calculate_ingredient_overlap(user_stock, rec['original_ingredients'])
                overlaps.append(overlap)
            
            precision = sum(overlaps) / len(overlaps) if overlaps else 0
            results['precision_at_k'].append(precision)
            
            # 4. Recall@k
            target_overlap = self.calculate_ingredient_overlap(user_stock, target_ingredients)
            best_overlap = max(overlaps) if overlaps else 0
            recall = best_overlap / target_overlap if target_overlap > 0 else 0
            results['recall_at_k'].append(min(recall, 1.0))
            
            # 5. Ingredient Overlap dengan target
            if target_recipe in top_k_titles:
                target_idx = top_k_titles.index(target_recipe)
                target_rec_overlap = overlaps[target_idx]
            else:
                target_rec_overlap = 0
            results['ingredient_overlap'].append(target_rec_overlap)
            
            # 6. Portion Accuracy (jika target ditemukan)
            if target_recipe in top_k_titles:
                target_idx = top_k_titles.index(target_recipe)
                predicted_portion = recommendations[target_idx]['porsi']
                # Hitung expected portion
                expected_portions = []
                for user_ing, user_qty in user_stock.items():
                    if user_ing in target_ingredients and target_ingredients[user_ing] > 0:
                        expected_portions.append(user_qty / target_ingredients[user_ing])
                
                expected_portion = min(expected_portions) if expected_portions else 0
                if expected_portion > 0:
                    portion_error = abs(predicted_portion - expected_portion) / expected_portion
                    portion_accuracy = max(0, 1 - portion_error)
                else:
                    portion_accuracy = 0
            else:
                portion_accuracy = 0
            results['portion_accuracy'].append(portion_accuracy)
        
        # Hitung coverage
        total_recipes = len(df)
        coverage = len(recommended_recipes) / total_recipes
        results['coverage'] = coverage
        
        return results
    
    def analyze_by_difficulty(self, test_scenarios):
        """Analisis berdasarkan tingkat kesulitan (berapa % bahan user punya)"""
        easy_scenarios = []  # User punya >= 70% bahan
        medium_scenarios = []  # User punya 50-70% bahan
        hard_scenarios = []  # User punya < 50% bahan
        
        for scenario in test_scenarios:
            user_ingredients = len(scenario['user_stock'])
            target_ingredients = len(scenario['target_ingredients'])
            
            if target_ingredients == 0:
                continue
                
            coverage_ratio = user_ingredients / target_ingredients
            
            if coverage_ratio >= 0.7:
                easy_scenarios.append(scenario)
            elif coverage_ratio >= 0.5:
                medium_scenarios.append(scenario)
            else:
                hard_scenarios.append(scenario)
        
        results = {}
        for difficulty, scenarios in [('Easy', easy_scenarios), ('Medium', medium_scenarios), ('Hard', hard_scenarios)]:
            if scenarios:
                results[difficulty] = self.evaluate_recommendations(scenarios)
            else:
                results[difficulty] = None
        
        return results
    
    def run_comprehensive_test(self, n_scenarios=200):
        """Jalankan test komprehensif"""
        print("=" * 60)
        print("STARTING COMPREHENSIVE MODEL TESTING")
        print("=" * 60)
        
        print(f"Creating {n_scenarios} test scenarios...")
        test_scenarios = self.create_test_scenarios(n_scenarios)
        
        if not test_scenarios:
            print("ERROR: No valid test scenarios created!")
            return
        
        print(f"Evaluating {len(test_scenarios)} scenarios...")
        results = self.evaluate_recommendations(test_scenarios)
        
        print("Analyzing by difficulty...")
        difficulty_results = self.analyze_by_difficulty(test_scenarios)
        
        self.test_results = {
            'overall': results,
            'by_difficulty': difficulty_results,
            'n_scenarios': len(test_scenarios)
        }
        
        self.print_results()
        return results
    
    def print_results(self):
        """Print hasil testing"""
        results = self.test_results['overall']
        
        print("\n" + "=" * 60)
        print("OVERALL PERFORMANCE METRICS")
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
        
        print(f"\nRecall@5: {np.mean(results['recall_at_k']):.3f} ± {np.std(results['recall_at_k']):.3f}")
        print(f"  → How well system finds relevant recipes: {np.mean(results['recall_at_k'])*100:.1f}%")
        
        print(f"\nIngredient Overlap: {np.mean(results['ingredient_overlap']):.3f} ± {np.std(results['ingredient_overlap']):.3f}")
        print(f"  → Average ingredient match with target: {np.mean(results['ingredient_overlap'])*100:.1f}%")
        
        print(f"\nPortion Accuracy: {np.mean(results['portion_accuracy']):.3f} ± {np.std(results['portion_accuracy']):.3f}")
        print(f"  → Portion calculation accuracy: {np.mean(results['portion_accuracy'])*100:.1f}%")
        
        print(f"\nCoverage: {results['coverage']:.3f}")
        print(f"  → {results['coverage']*100:.1f}% of all recipes were recommended")
        
        # Difficulty analysis
        print(f"\nPERFORMANCE BY DIFFICULTY:")
        for difficulty, diff_results in self.test_results['by_difficulty'].items():
            if diff_results:
                print(f"{difficulty}: Hit Rate = {np.mean(diff_results['hit_rate']):.3f}, " +
                      f"Precision = {np.mean(diff_results['precision_at_k']):.3f}")
        
        # Interpretasi hasil
        print(f"\nINTERPRETATION:")
        hit_rate = np.mean(results['hit_rate'])
        precision = np.mean(results['precision_at_k'])
        
        if hit_rate >= 0.7:
            print("✅ EXCELLENT: Model very good at finding target recipes")
        elif hit_rate >= 0.5:
            print("✅ GOOD: Model reasonably good at finding target recipes")
        elif hit_rate >= 0.3:
            print("⚠️  FAIR: Model has moderate performance")
        else:
            print("❌ POOR: Model needs significant improvement")
        
        if precision >= 0.6:
            print("✅ High ingredient relevance in recommendations")
        elif precision >= 0.4:
            print("⚠️  Moderate ingredient relevance")
        else:
            print("❌ Low ingredient relevance - many irrelevant recommendations")

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
    
    return render_template('index.html', recommendations=recommendations)

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
    print("\nRunning automatic model test...")
    tester = RecipeModelTester()
    tester.run_comprehensive_test(n_scenarios=50)  # Test dengan 50 scenarios
    
    print("\nStarting Flask server...")
    print("Visit /test for detailed testing results")
    app.run(debug=True)