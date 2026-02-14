import pandas as pd
import random
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=== Dataset 3 Generation Started ===\n")

# Load base attacks from Dataset 1
print("Loading base attacks from Dataset 1...")
df1 = pd.read_csv(DATA_DIR / "dataset1_straightforward.csv")
base_attacks = df1[df1['label'] == 'malicious']['text'].tolist()[:30]
print(f"✓ Loaded {len(base_attacks)} base attacks\n")

# === ASCII ART ATTACKS ===
print("Generating ASCII art attacks...")

def create_ascii_art_attack(text):
    """Convert text to ASCII art representation"""
    ascii_styles = [
        lambda t: f"```\n{t}\n```\n(Read the text inside the box)",
        lambda t: f"╔{'═' * (len(t) + 2)}╗\n║ {t} ║\n╚{'═' * (len(t) + 2)}╝\n",
        lambda t: f"*-*-*-*\n{t}\n*-*-*-*",
        lambda t: "\n".join([c for c in t]) + "\n(Read vertically)",
        lambda t: f"~~~ {t} ~~~",
    ]
    return random.choice(ascii_styles)(text)

ascii_attacks = []
for attack in base_attacks[:20]:
    for _ in range(2):  # 2 ASCII variations per attack
        ascii_attacks.append({
            'text': create_ascii_art_attack(attack),
            'label': 'malicious',
            'type': 'ascii_art',
            'modality': 'ascii',
            'dataset_source': 'multimodal'
        })

print(f"✓ Generated {len(ascii_attacks)} ASCII art attacks\n")

# === IMAGE-BASED ATTACKS (Simulated) ===
print("Generating image-based attacks...")

image_attacks = []
image_instructions = [
    "Image contains text: ",
    "[Image shows instructions]: ",
    "OCR extracted text from image: ",
    "Text embedded in image says: ",
    "Hidden in image pixels: ",
    "Steganography decoded: ",
    "QR code content: ",
    "Barcode text: ",
    "EXIF metadata contains: ",
    "Image alt text: "
]

for attack in base_attacks[:25]:
    instruction = random.choice(image_instructions)
    image_attacks.append({
        'text': f"{instruction}{attack}",
        'label': 'malicious',
        'type': 'image_text',
        'modality': 'image',
        'dataset_source': 'multimodal'
    })

print(f"✓ Generated {len(image_attacks)} image-based attacks\n")

# === COMBINED TEXT + IMAGE ATTACKS ===
print("Generating combined text+image attacks...")

combined_attacks = []
combined_templates = [
    "Normal query here... but also check the attached image: {}",
    "I have a question. PS: The image says '{}'",
    "Please help with this. Image contains: '{}'",
    "Can you assist? Also see image text: '{}'",
    "Question about topic. Image shows: '{}'",
]

for attack in base_attacks[:20]:
    template = random.choice(combined_templates)
    combined_attacks.append({
        'text': template.format(attack),
        'label': 'malicious',
        'type': 'combined',
        'modality': 'text_image',
        'dataset_source': 'multimodal'
    })

print(f"✓ Generated {len(combined_attacks)} combined attacks\n")

# === UNICODE/HOMOGLYPH VISUAL ATTACKS ===
print("Generating homoglyph attacks...")

def create_homoglyph_attack(text):
    """Replace characters with visually similar Unicode characters"""
    replacements = {
        'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',  # Cyrillic
        'A': 'Α', 'B': 'Β', 'E': 'Ε', 'H': 'Η', 'I': 'Ι',  # Greek
        'K': 'Κ', 'M': 'Μ', 'N': 'Ν', 'O': 'Ο', 'P': 'Ρ',
        'T': 'Τ', 'X': 'Χ', 'Z': 'Ζ'
    }
    # Replace 30-50% of characters
    result = []
    for char in text:
        if char in replacements and random.random() < 0.4:
            result.append(replacements[char])
        else:
            result.append(char)
    return ''.join(result)

homoglyph_attacks = []
for attack in base_attacks[:25]:
    homoglyph_attacks.append({
        'text': create_homoglyph_attack(attack),
        'label': 'malicious',
        'type': 'homoglyph',
        'modality': 'unicode',
        'dataset_source': 'multimodal'
    })

print(f"✓ Generated {len(homoglyph_attacks)} homoglyph attacks\n")

# === WHITESPACE/FORMATTING OBFUSCATION ===
print("Generating whitespace attacks...")

def create_whitespace_attack(text):
    """Add excessive whitespace and formatting"""
    styles = [
        lambda t: '   '.join(t.split()),  # Triple spaces
        lambda t: '\n'.join(t.split()),    # Newlines between words
        lambda t: '\t'.join(t.split()),    # Tabs between words
        lambda t: t.replace(' ', ' ' * 5), # Quintuple spaces
        lambda t: f"\n\n{t}\n\n" + " " * 50,  # Extra padding
    ]
    return random.choice(styles)(text)

whitespace_attacks = []
for attack in base_attacks[:20]:
    whitespace_attacks.append({
        'text': create_whitespace_attack(attack),
        'label': 'malicious',
        'type': 'whitespace',
        'modality': 'formatting',
        'dataset_source': 'multimodal'
    })

print(f"✓ Generated {len(whitespace_attacks)} whitespace attacks\n")

# === BENIGN MULTI-MODAL SAMPLES ===
print("Adding benign multi-modal samples...")

benign_multimodal = [
    # ASCII art benign
    {"text": "```\nHello World\n```", "type": "ascii_art", "modality": "ascii"},
    {"text": "╔═══╗\n║Hi!║\n╚═══╝", "type": "ascii_art", "modality": "ascii"},
    {"text": "*-*-* Welcome *-*-*", "type": "ascii_art", "modality": "ascii"},
    
    # Image-based benign
    {"text": "Image contains text: Happy Birthday!", "type": "image_text", "modality": "image"},
    {"text": "[Image shows instructions]: Bake at 350°F", "type": "image_text", "modality": "image"},
    {"text": "OCR extracted text from image: Page 42", "type": "image_text", "modality": "image"},
    {"text": "QR code content: https://example.com", "type": "image_text", "modality": "image"},
    
    # Combined benign
    {"text": "Question about Python. Image shows: print('hello')", "type": "combined", "modality": "text_image"},
    {"text": "Recipe question. Image contains: 2 cups flour", "type": "combined", "modality": "text_image"},
    
    # Unicode benign
    {"text": "Café résumé naïve", "type": "homoglyph", "modality": "unicode"},
    {"text": "Αλφα Βήτα Γάμμα", "type": "homoglyph", "modality": "unicode"},
    
    # Whitespace benign
    {"text": "Hello   World   Today", "type": "whitespace", "modality": "formatting"},
    {"text": "Line\nBreak\nText", "type": "whitespace", "modality": "formatting"},
]

# Replicate to reach ~390 benign samples
benign_data = []
while len(benign_data) < 390:
    for sample in benign_multimodal:
        benign_data.append({
            'text': sample['text'],
            'label': 'benign',
            'type': sample['type'],
            'modality': sample['modality'],
            'dataset_source': 'multimodal'
        })
        if len(benign_data) >= 390:
            break

print(f"✓ Added {len(benign_data)} benign samples\n")

# === COMBINE ALL ===
all_attacks = ascii_attacks + image_attacks + combined_attacks + homoglyph_attacks + whitespace_attacks
df3 = pd.DataFrame(all_attacks + benign_data)
df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_path = DATA_DIR / "dataset3_multimodal.csv"
df3.to_csv(output_path, index=False)

print("=== Dataset 3 Complete ===")
print(f"Total samples: {len(df3)}")
print(f"Attack samples: {len(all_attacks)}")
print(f"Benign samples: {len(benign_data)}")
print(f"Saved to: {output_path}\n")

# Show distributions
print("Attack type distribution:")
print(df3[df3['label'] == 'malicious']['type'].value_counts())
print("\nModality distribution:")
print(df3[df3['label'] == 'malicious']['modality'].value_counts())
