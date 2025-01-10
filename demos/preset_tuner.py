import json
import numpy as np
import matplotlib.pyplot as plt
import random
import string

from sklearn.ensemble import RandomForestRegressor



def presets_to_feature_array(presets, sentinel=-1.0):
    """
    Transform a list of preset dicts into a NumPy feature matrix.
    
    The columns are (in order):
      0) sampler_mirostat (0/1)
      1) sampler_top_p    (0/1)
      2) sampler_top_k    (0/1)
      3) mirostat_eta
      4) mirostat_tau_0
      5) mirostat_tau_1
      6) mirostat_tau_2
      7) mirostat_tau_3
      8) top_p
      9) top_k
     10) temperature
     11) cfg_coef

    Parameters
    ----------
    presets : list of dict
        Each dict is one preset with keys like:
          - "mirostat_eta", "mirostat_tau", "temperature", "cfg_coef"
          - "top_p" or "top_k"
          - "name" (ignored here, except for reference)
        Example:
          {
            'name': 'top_p draft',
            'top_p': 0.2,
            'temperature': 1.2,
            'cfg_coef': 3.0
          }
    sentinel : float
        The value used for missing parameters in each preset.

    Returns
    -------
    np.ndarray
        2D array of shape (N, 12), where N = len(presets).
    """
    
    feature_list = []
    
    for preset in presets:
        # 1) Identify which sampler type we have.
        #    We'll do a one-hot: (is_mirostat, is_top_p, is_top_k).
        is_mirostat = 1 if "mirostat_eta" in preset else 0
        is_top_p    = 1 if "top_p" in preset else 0
        is_top_k    = 1 if "top_k" in preset else 0
        
        # 2) Handle mirostat parameters.
        mirostat_eta = preset["mirostat_eta"] if "mirostat_eta" in preset else sentinel
        
        # 'mirostat_tau' is a list of 4 floats in your data. We'll expand them into 4 separate columns.
        if "mirostat_tau" in preset and isinstance(preset["mirostat_tau"], list):
            tau_vals = preset["mirostat_tau"]
            # If you want exactly 4 entries, ensure we pick tau_vals[0..3] or sentinel if missing
            if len(tau_vals) < 4:
                # Pad if fewer than 4, or handle as needed
                tau_vals = tau_vals + [sentinel]*(4 - len(tau_vals))
            mirostat_tau_0, mirostat_tau_1, mirostat_tau_2, mirostat_tau_3 = tau_vals[:4]
        else:
            # If not present, set all to sentinel
            mirostat_tau_0 = mirostat_tau_1 = mirostat_tau_2 = mirostat_tau_3 = sentinel
        
        # 3) If it's a 'top_p' preset, fill top_p; otherwise sentinel.
        top_p_val = preset["top_p"] if "top_p" in preset else sentinel
        
        # 4) If it's a 'top_k' preset, fill top_k; otherwise sentinel.
        top_k_val = preset["top_k"] if "top_k" in preset else sentinel
        
        # 5) temperature and cfg_coef could appear in any preset, or use sentinel if absent.
        temperature = preset["temperature"] if "temperature" in preset else sentinel
        cfg_coef    = preset["cfg_coef"]    if "cfg_coef"    in preset else sentinel
        
        # 6) Construct the feature vector for this preset
        feature_vector = [
            is_mirostat,
            is_top_p,
            is_top_k,
            mirostat_eta,
            mirostat_tau_0,
            mirostat_tau_1,
            mirostat_tau_2,
            mirostat_tau_3,
            top_p_val,
            top_k_val,
            temperature,
            cfg_coef
        ]
        
        feature_list.append(feature_vector)
    
    # Convert to NumPy array
    return np.array(feature_list, dtype=float)

def sample_presets(N, random_seed=42):
    """
    Generate N new preset dictionaries following conditional rules:
      - If sampler_type = 'mirostat':
          keys: mirostat_eta, mirostat_tau (list of 4), temperature, cfg_coef
      - If sampler_type = 'top_p':
          keys: top_p, temperature, cfg_coef
      - If sampler_type = 'top_k':
          keys: top_k, temperature, cfg_coef

    Each preset gets a more unique random name by combining:
       (adjective + noun + sampler) + '_' + random 3-char suffix

    This significantly reduces naming collisions.

    Returns
    -------
    list of dict
        Each dict is a preset with the appropriate fields, including a random name.
    """

    random.seed(random_seed)

    # Possible sampler types
    sampler_types = ["mirostat", "top_p", "top_k"]

    # Larger lists to reduce collisions
    adjectives = [
        "Mysterious", "Melodic", "Dancing", "Crazy", "Zany", "Harmonic", "Smooth", "Spicy", 
        "Euphoric", "Eclectic", "Electric", "Hypnotic", "Glittering", "Lyrical", "Soaring", 
        "Ambient", "Cinematic", "Trippy", "Subtle", "Quirky", "Mystic", "Ethereal", "Funky", 
        "Groovy", "Velvet", "Velociraptor", "Epic", "Uplifting", "Seductive", "Galactic"
    ]
    nouns = [
        "Tiger", "Penguin", "Dragon", "Fox", "Butterfly", "Rhino", "Mermaid", "Robot",
        "Phoenix", "Cactus", "Unicorn", "Camel", "Leopard", "Jellyfish", "Orchid", 
        "Kitten", "Eagle", "Chameleon", "Monkey", "Zebra", "Pelican", "Wombat", "Alligator",
        "Octopus", "Nightmare", "Shark", "Dolphin", "Hedgehog", "Koala", "Hamster"
    ]

    def random_name_for_sampler(sampler):
        """Generate a random, more collision-resistant name for the sampler."""
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        sampler_label = sampler.capitalize()
        # Add a short random suffix (3 alphanumeric chars)
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
        # Example: "MysteriousTigerMirostat_n4g"
        return f"{adj}{noun}{sampler_label}_{suffix}"

    sampled_presets = []
    for _ in range(N):
        sampler_type = random.choice(sampler_types)
        preset_name = random_name_for_sampler(sampler_type)

        if sampler_type == "mirostat":
            eta = random.uniform(0.01, 0.2)
            tau_list = [random.uniform(1.0, 4.0) for _ in range(4)]
            temperature = random.uniform(0.8, 2.0)
            cfg_coef = random.uniform(2.0, 4.0)  # or fix at 3.0 if desired

            preset_dict = {
                "name": preset_name,
                "mirostat_eta": eta,
                "mirostat_tau": tau_list,
                "temperature": temperature,
                "cfg_coef": cfg_coef
            }

        elif sampler_type == "top_p":
            p_value = random.uniform(0.1, 0.9)
            temperature = random.uniform(0.8, 2.0)
            cfg_coef = random.uniform(2.0, 4.0)

            preset_dict = {
                "name": preset_name,
                "top_p": p_value,
                "temperature": temperature,
                "cfg_coef": cfg_coef
            }

        else:  # sampler_type == "top_k"
            k_value = random.randint(10, 400)  # integer top_k
            temperature = random.uniform(0.8, 2.0)
            cfg_coef = random.uniform(2.0, 4.0)

            preset_dict = {
                "name": preset_name,
                "top_k": k_value,
                "temperature": temperature,
                "cfg_coef": cfg_coef
            }

        sampled_presets.append(preset_dict)

    return sampled_presets


def suggest_preset(presets, preset_ratings, ucb_alpha=1., n_random_presets=1000):
    target = [preset_ratings[preset['name']] for preset in presets]
    feature_array = presets_to_feature_array(presets)
    
    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(feature_array, target)

    random_presets = sample_presets(n_random_presets)
    random_presets_array = presets_to_feature_array(random_presets)

    estimators_prediction_array = np.array([est.predict(random_presets_array) for est in rf.estimators_]).transpose()
    elo_estimations = estimators_prediction_array.mean(1)
    elo_estimations_std = estimators_prediction_array.std(1)
    ucb_values = elo_estimations+ucb_alpha*elo_estimations_std

    return random_presets[ucb_values.argmax()]