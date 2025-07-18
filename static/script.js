// Recipe Recommender JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // Initialize form functionality
    initializeForm();

    // Auto-scroll to results if they exist
    const results = document.querySelector('.results-section');
    if (results) {
        setTimeout(() => {
            results.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
});

function initializeForm() {
    // Validate main ingredient is required
    const mainIngredientInput = document.querySelector('.main-ingredient-input');
    if (mainIngredientInput) {
        mainIngredientInput.addEventListener('input', function () {
            validateMainIngredient();
        });
    }

    // Add event listeners to existing inputs
    const ingredientInputs = document.querySelectorAll('.ingredient-name');
    ingredientInputs.forEach(input => {
        input.addEventListener('input', function () {
            validateIngredientName(this);
        });
    });

    const quantityInputs = document.querySelectorAll('.ingredient-qty');
    quantityInputs.forEach(input => {
        input.addEventListener('input', function () {
            validateQuantity(this);
        });
    });

    // Setup auto-suggest
    setupAutoSuggest();
}

function addIngredient() {
    const additionalIngredients = document.querySelector('.additional-ingredients');

    // Create new ingredient row
    const newRow = document.createElement('div');
    newRow.className = 'ingredient-row';

    newRow.innerHTML = `
        <input type="text" name="ingredient_name[]" placeholder="Contoh: bawang, garam, cabai" 
               class="ingredient-name">
        <input type="number" name="ingredient_qty[]" placeholder="Jumlah (gram)" 
               class="ingredient-qty" step="0.01">
        <span class="ingredient-unit">gram</span>
        <button type="button" class="remove-btn" onclick="removeIngredient(this)">✕</button>
    `;

    additionalIngredients.appendChild(newRow);

    // Add event listeners to new inputs
    const newIngredientInput = newRow.querySelector('.ingredient-name');
    const newQuantityInput = newRow.querySelector('.ingredient-qty');

    newIngredientInput.addEventListener('input', function () {
        validateIngredientName(this);
    });

    newQuantityInput.addEventListener('input', function () {
        validateQuantity(this);
    });

    newIngredientInput.addEventListener('focus', function () {
        setupAutoSuggest();
    });

    // Focus on new ingredient input
    newIngredientInput.focus();

    // Add animation
    newRow.style.opacity = '0';
    newRow.style.transform = 'translateY(-10px)';

    setTimeout(() => {
        newRow.style.transition = 'all 0.3s ease';
        newRow.style.opacity = '1';
        newRow.style.transform = 'translateY(0)';
    }, 10);
}

function removeIngredient(button) {
    const row = button.closest('.ingredient-row');
    const additionalIngredients = document.querySelector('.additional-ingredients');

    // Only remove if there's more than one additional ingredient row
    if (additionalIngredients.children.length > 1) {
        // Add animation
        row.style.transition = 'all 0.3s ease';
        row.style.opacity = '0';
        row.style.transform = 'translateY(-10px)';

        setTimeout(() => {
            row.remove();
        }, 300);
    } else {
        // If it's the last row, just clear the inputs
        const nameInput = row.querySelector('.ingredient-name');
        const qtyInput = row.querySelector('.ingredient-qty');

        nameInput.value = '';
        qtyInput.value = '';

        // Add shake animation
        row.style.animation = 'shake 0.5s ease-in-out';
        setTimeout(() => {
            row.style.animation = '';
        }, 500);
    }
}

function validateMainIngredient() {
    const mainInput = document.querySelector('.main-ingredient-input');
    const submitBtn = document.querySelector('.submit-btn');

    if (mainInput.value.trim() === '') {
        mainInput.style.borderColor = '#e74c3c';
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.6';
        showValidationMessage(mainInput, 'Bahan utama harus diisi');
    } else {
        mainInput.style.borderColor = '#f39c12';
        submitBtn.disabled = false;
        submitBtn.style.opacity = '1';
        hideValidationMessage(mainInput);
    }
}

function validateIngredientName(input) {
    const value = input.value.trim();

    if (value.length > 0 && value.length < 2) {
        input.style.borderColor = '#e74c3c';
        showValidationMessage(input, 'Nama bahan minimal 2 karakter');
    } else if (value.length > 50) {
        input.style.borderColor = '#e74c3c';
        showValidationMessage(input, 'Nama bahan maksimal 50 karakter');
    } else {
        input.style.borderColor = '#4facfe';
        hideValidationMessage(input);
    }
}

function validateQuantity(input) {
    const value = parseFloat(input.value);

    if (input.value !== '' && (isNaN(value) || value <= 0)) {
        input.style.borderColor = '#e74c3c';
        showValidationMessage(input, 'Jumlah harus berupa angka positif');
    } else if (value > 10000) {
        input.style.borderColor = '#e74c3c';
        showValidationMessage(input, 'Jumlah maksimal 10kg (10000g)');
    } else {
        input.style.borderColor = '#4facfe';
        hideValidationMessage(input);
    }
}

function showValidationMessage(input, message) {
    // Remove existing message
    hideValidationMessage(input);

    const messageDiv = document.createElement('div');
    messageDiv.className = 'validation-message';
    messageDiv.textContent = message;
    messageDiv.style.cssText = `
        color: #e74c3c;
        font-size: 12px;
        margin-top: 5px;
        padding: 5px;
        background: #fdf2f2;
        border-radius: 4px;
        border: 1px solid #fecaca;
    `;

    input.parentNode.appendChild(messageDiv);
}

function hideValidationMessage(input) {
    const existingMessage = input.parentNode.querySelector('.validation-message');
    if (existingMessage) {
        existingMessage.remove();
    }
}

// Common ingredients for auto-suggest
const commonIngredients = [
    'ayam', 'daging sapi', 'ikan', 'udang', 'tahu', 'tempe',
    'beras', 'mie', 'pasta', 'kentang', 'wortel', 'brokoli',
    'bawang merah', 'bawang putih', 'cabai', 'tomat', 'timun',
    'garam', 'gula', 'minyak', 'mentega', 'santan', 'kecap',
    'tepung terigu', 'tepung beras', 'telur', 'susu', 'keju'
];

function setupAutoSuggest() {
    const ingredientInputs = document.querySelectorAll('.ingredient-name');

    ingredientInputs.forEach(input => {
        input.addEventListener('input', function () {
            const value = this.value.toLowerCase();

            if (value.length >= 2) {
                const suggestions = commonIngredients.filter(ingredient =>
                    ingredient.toLowerCase().includes(value)
                );
                showSuggestions(this, suggestions);
            } else {
                hideSuggestions(this);
            }
        });

        input.addEventListener('blur', function () {
            // Hide suggestions after a delay to allow clicking
            setTimeout(() => hideSuggestions(this), 200);
        });

        input.addEventListener('focus', function () {
            if (this.value.length >= 2) {
                const suggestions = commonIngredients.filter(ingredient =>
                    ingredient.toLowerCase().includes(this.value.toLowerCase())
                );
                showSuggestions(this, suggestions);
            }
        });
    });
}

function showSuggestions(input, suggestions) {
    hideSuggestions(input);

    if (suggestions.length === 0) return;

    const rect = input.getBoundingClientRect();
    const dropdown = document.createElement('div');
    dropdown.className = 'suggestions-dropdown';
    dropdown.style.position = 'absolute';
    dropdown.style.top = `${rect.bottom + window.scrollY}px`;
    dropdown.style.left = `${rect.left + window.scrollX}px`;
    dropdown.style.width = `${rect.width}px`;
    dropdown.style.backgroundColor = 'white';
    dropdown.style.border = '1px solid #ccc';
    dropdown.style.borderRadius = '4px';
    dropdown.style.zIndex = '1000';
    dropdown.style.maxHeight = '200px';
    dropdown.style.overflowY = 'auto';
    dropdown.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
    dropdown.style.padding = '5px 0';

    suggestions.forEach(suggestion => {
        const item = document.createElement('div');
        item.textContent = suggestion;
        item.style.padding = '8px 12px';
        item.style.cursor = 'pointer';
        item.style.transition = 'background-color 0.2s';

        item.addEventListener('mouseenter', () => {
            item.style.backgroundColor = '#f0f0f0';
        });

        item.addEventListener('mouseleave', () => {
            item.style.backgroundColor = 'white';
        });

        item.addEventListener('click', () => {
            input.value = suggestion;
            hideSuggestions(input);
            validateIngredientName(input);
        });

        dropdown.appendChild(item);
    });

    document.body.appendChild(dropdown);
}

function hideSuggestions(input) {
    const existing = document.querySelector('.suggestions-dropdown');
    if (existing) {
        existing.remove();
    }
}