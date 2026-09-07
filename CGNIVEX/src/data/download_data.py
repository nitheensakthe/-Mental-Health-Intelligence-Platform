import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "mental_health_dataset.csv")

def generate_synthetic_dataset(num_samples=2800):
    """Generates a high-quality multi-class mental health dataset for development and benchmarking."""
    np.random.seed(42)
    categories = ['Normal', 'Depression', 'Anxiety', 'Stress', 'Suicidal', 'Bi-Polar', 'Personality Disorder']
    
    templates = {
        'Normal': [
            "Had a great day spending time with family and friends at the park.",
            "Really enjoying learning python programming and data science lately.",
            "Just finished reading a fantastic book on history and science.",
            "Feeling grateful for good health and peaceful sunny weather today.",
            "Cooking dinner with loved ones and watching a pleasant movie tonight.",
            "Completed my workout session and feel energized for the upcoming week.",
            "Looking forward to the weekend getaway with my university roommates.",
            "Everything is going smoothly at work, accomplished all key goals.",
            "Tried a new coffee blend today and it was surprisingly delicious.",
            "Peaceful Sunday morning relaxing with fresh tea and acoustic music."
        ],
        'Depression': [
            "Feeling completely empty inside, nothing seems to bring joy anymore.",
            "Struggling to get out of bed every morning, everything feels overwhelming.",
            "Tired of pretending to be okay when I feel so alone and hollow.",
            "Lost interest in all my hobbies, just feel numb and detached.",
            "Constant feeling of sadness and hopelessness that won't go away.",
            "Why does life feel so exhausting and pointless every single day?",
            "Isolating myself from friends because I don't have the energy to talk.",
            "Dark thoughts keep creeping in whenever I am left alone in silence.",
            "Feel like a burden to everyone around me, nobody really understands.",
            "Waking up feeling heavy and empty, waiting for the day to just end."
        ],
        'Anxiety': [
            "My heart is racing fast for no apparent reason, panic is setting in.",
            "Cannot stop overthinking every minor detail, constant dread.",
            "Feel extremely nervous about tomorrow's meeting, hands are shaking.",
            "Worried about the future all the time, my mind just won't rest.",
            "Sudden wave of panic made it hard to breathe properly in public.",
            "Restless thoughts keeping me awake all night long, constant fear.",
            "Terrified of making mistakes and failing in front of everybody.",
            "Tightness in my chest and constant feeling that something terrible will happen.",
            "Social situations make me feel overwhelmed and deeply uncomfortable.",
            "Overanalyzing past conversations over and over again in my head."
        ],
        'Stress': [
            "Deadlines piling up rapidly and I am drowning under immense pressure.",
            "Working 14 hours a day with zero rest, completely burned out.",
            "Too many responsibilities at once, feeling extremely tense and irritable.",
            "Cannot sleep properly due to overwhelming work pressure and exams.",
            "Headaches every day from constant tension and unrelenting strain.",
            "Juggling multiple projects without any support is breaking me down.",
            "So stressed out that I can barely concentrate on basic tasks.",
            "Exhausted physically and mentally from non-stop high-pressure environment.",
            "Feel like I am going to explode if one more thing goes wrong today.",
            "No time for self-care or relaxation, constantly running on fumes."
        ],
        'Suicidal': [
            "Feeling like giving up on everything, cannot carry this heavy pain.",
            "Wishing I could just disappear and never wake up ever again.",
            "The mental agony is unbearable, don't know how much longer I can endure.",
            "Searching for a way out of this endless darkness and hopeless agony.",
            "Feel like everyone would be better off without me existing around.",
            "Reached my absolute limit, losing all hope for any recovery or peace.",
            "I just want the noise in my head to stop permanently.",
            "Feeling utterly defeated with zero reasons to keep fighting anymore.",
            "Goodbyes feel closer than ever, I am completely broken beyond repair.",
            "Deep inescapable dark pit, desperate for an end to this torment."
        ],
        'Bi-Polar': [
            "One day I feel invincible with unlimited energy, next day I can't leave bed.",
            "Extreme mood swings are ruining my life, going from high euphoria to deep depression.",
            "Feeling hyperactive and spending money impulsively, then sudden dark crash.",
            "Racing thoughts and bursts of chaotic creativity followed by total paralysis.",
            "Cannot control these sudden drastic shifts in my emotions and energy levels.",
            "Spent 48 hours awake feeling on top of the world, now completely hopeless.",
            "Unpredictable emotional waves driving away the people who care about me.",
            "Manic episode made me feel unstoppable, now dealing with severe aftermath.",
            "Swinging between intense enthusiasm and devastating deep despair rapidly.",
            "Struggling to find stability when my mind switches extremes constantly."
        ],
        'Personality Disorder': [
            "Fear of abandonment makes me push people away before they leave me.",
            "Intense unstable relationships, loving someone one minute and hating them next.",
            "Unsure of who I really am, identity feels fragmented and empty constantly.",
            "Impulsive behavior and sudden anger outbursts that I deeply regret after.",
            "Difficulty maintaining stable connections due to emotional turbulence.",
            "Feeling disconnected from reality and my own sense of identity.",
            "Rapidly shifting perceptions of self and others, struggle with trust.",
            "Emotional reactions feel completely out of proportion to situations.",
            "Constantly feeling empty inside and seeking external validation desperately.",
            "Volatile mood shifts and chronic fear of rejection dominating my daily life."
        ]
    }
    
    data = []
    for category in categories:
        samples_per_cat = num_samples // len(categories)
        cat_templates = templates[category]
        for i in range(samples_per_cat):
            base_text = np.random.choice(cat_templates)
            prefix = np.random.choice(["", "Honestly, ", "Right now, ", "I feel like ", "Lately, ", "Today: "])
            suffix = np.random.choice(["", " #mentalhealth", " Need advice.", " Anyone else feel this?", " Sigh.", " ..."])
            text = f"{prefix}{base_text}{suffix}".strip()
            data.append({"text": text, "status": category})
            
    df = pd.DataFrame(data)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset successfully created at '{OUTPUT_FILE}' with {len(df)} rows across 7 classes.")
    return df

if __name__ == "__main__":
    generate_synthetic_dataset()
