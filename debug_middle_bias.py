#!/usr/bin/env python3
"""
Demonstrate why the old confidence system created middle bias
"""

def old_system_complete(rating, confidence):
    """The complete old system with confidence adjustment + rounding"""
    # Apply confidence adjustment
    if confidence < 0.9:
        uncertainty_factor = (0.9 - confidence) / 0.9
        if rating <= 1.0:
            if rating < 0.5:
                adjustment = (0.5 - rating) * uncertainty_factor * 0.3
                rating = rating + adjustment
            else:
                adjustment = (1.0 - rating) * uncertainty_factor * 0.4
                rating = rating + adjustment
        elif rating > 1.0:
            if rating > 1.5:
                adjustment = (rating - 1.5) * uncertainty_factor * 0.3
                rating = rating - adjustment
            else:
                adjustment = (1.5 - rating) * uncertainty_factor * 0.4
                rating = rating + adjustment
    
    # Then round to discrete values (this was the problem!)
    if rating < 0.4:
        return 0.0
    elif rating < 1.4:
        return 1.0  # Most things ended up here!
    else:
        return 2.0

def new_system_complete(rating, confidence):
    """New system: no confidence adjustment, direct classification"""
    # Round rating to discrete values immediately
    if rating < 0.4:
        return 0.0
    elif rating < 1.4:
        return 1.0
    else:
        return 2.0

# Test both systems
examples = [
    (0.0, 0.5), (0.2, 0.6), (0.5, 0.7), (0.8, 0.5),
    (1.0, 0.6), (1.2, 0.7), (1.5, 0.5), (1.8, 0.6), (2.0, 0.5)
]

print('COMPARISON: Old vs New System')
print('=' * 50)
print('Original → Old Result → New Result')

old_uncertain = 0
new_uncertain = 0

for original, conf in examples:
    old_final = old_system_complete(original, conf)
    new_final = new_system_complete(original, conf)
    
    if old_final == 1.0:
        old_uncertain += 1
    if new_final == 1.0:
        new_uncertain += 1
    
    print(f'{original:3.1f} (conf={conf}) → {old_final:3.1f} → {new_final:3.1f}')

print(f'\nSUMMARY:')
print(f'Old system: {old_uncertain}/{len(examples)} became uncertain (1.0) = {old_uncertain/len(examples)*100:.1f}%')
print(f'New system: {new_uncertain}/{len(examples)} became uncertain (1.0) = {new_uncertain/len(examples)*100:.1f}%')

print(f'\nWHY THE OLD SYSTEM FAILED:')
print(f'1. Confidence < 0.9 triggered adjustment (most cases)')
print(f'2. Adjustment always pushed toward middle ranges')
print(f'3. Rounding boundaries (0.4-1.4) captured most adjusted values')
print(f'4. Result: Everything became "uncertain" regardless of content')