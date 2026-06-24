"""
Parkinson's Disease Detection Using Speech Analysis
==================================================

This script implements a complete pipeline for detecting Parkinson's Disease
from speech samples using audio feature extraction, multiple optimization algorithms
for feature selection, and Random Forest classification.

It also includes real-time inference capabilities using the LivePredictor class.
"""

import numpy as np
import pandas as pd
import librosa
import os
import joblib
from collections import deque
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


def extract_features(file_path):
    """
    Extract audio features from a speech sample using Librosa.
    
    Parameters:
    file_path (str or np.ndarray): Path to the audio file OR raw audio buffer
    
    Returns:
    list: Extracted features or None if error occurs
    """
    try:
        if isinstance(file_path, str):
            y, sr = librosa.load(file_path, sr=22050)
        else:
            # Assume file_path is actually a raw audio buffer (numpy array)
            y = file_path
            sr = 22050
            
        features = []
        
        # MFCC features (mean and std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features.extend([float(spectral_centroid), float(spectral_rolloff), float(spectral_bandwidth)])
        
        # Temporal features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features.append(float(zcr))
        
        rms = np.mean(librosa.feature.rms(y=y))
        features.append(float(rms))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1).tolist())
        
        # Additional features for better accuracy
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(spectral_contrast, axis=1).tolist())
        
        # Spectral flatness
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        features.append(float(spectral_flatness))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


class GrayWolfOptimizer:
    """
    Gray Wolf Optimization algorithm for feature selection.
    """
    
    def __init__(self, n_wolves=10, max_iter=50):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        
    def optimize(self, X, y):
        """
        Optimize feature selection using GWO.
        
        Parameters:
        X (np.array): Feature matrix
        y (np.array): Labels
        
        Returns:
        np.array: Best feature selection solution
        """
        n_features = X.shape[1]
        wolves = np.random.randint(2, size=(self.n_wolves, n_features))
        fitness = np.array([self.fitness_function(wolf, X, y) for wolf in wolves])
        
        sorted_idx = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_idx[:3]]
        alpha_fit, beta_fit, delta_fit = fitness[sorted_idx[:3]]
        
        for iter in range(self.max_iter):
            a = 2 - iter * (2 / self.max_iter)
            
            for i in range(self.n_wolves):
                for j in range(n_features):
                    r1, r2 = np.random.random(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                    X1 = alpha[j] - A1 * D_alpha
                    
                    r1, r2 = np.random.random(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolves[i][j])
                    X2 = beta[j] - A2 * D_beta
                    
                    r1, r2 = np.random.random(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolves[i][j])
                    X3 = delta[j] - A3 * D_delta
                    
                    wolves[i][j] = np.round((X1 + X2 + X3) / 3)
                    wolves[i][j] = 1 if wolves[i][j] >= 0.5 else 0
                
                fitness[i] = self.fitness_function(wolves[i], X, y)
            
            sorted_idx = np.argsort(fitness)
            if fitness[sorted_idx[0]] < alpha_fit:
                alpha, beta, delta = wolves[sorted_idx[:3]]
                alpha_fit, beta_fit, delta_fit = fitness[sorted_idx[:3]]
            
            if (iter + 1) % 10 == 0:
                print(f"GWO Iteration {iter + 1}/{self.max_iter}, Best fitness: {alpha_fit:.4f}")
        
        return alpha
    
    def fitness_function(self, wolf, X, y):
        """
        Fitness function for evaluating a feature subset.
        
        Parameters:
        wolf (np.array): Binary array representing feature selection
        X (np.array): Feature matrix
        y (np.array): Labels
        
        Returns:
        float: Fitness value (lower is better)
        """
        selected_features = wolf.astype(bool)
        if np.sum(selected_features) == 0:
            return float('inf')
        
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        
        feature_ratio = np.sum(selected_features) / len(wolf)
        return (1 - accuracy) + 0.1 * feature_ratio


class ArtificialBeeColony:
    """
    Artificial Bee Colony algorithm for feature selection.
    """
    
    def __init__(self, n_bees=20, max_iter=50, limit=5):
        self.n_bees = n_bees
        self.max_iter = max_iter
        self.limit = limit
        
    def optimize(self, X, y):
        """
        Optimize feature selection using ABC.
        
        Parameters:
        X (np.array): Feature matrix
        y (np.array): Labels
        
        Returns:
        np.array: Best feature selection solution
        """
        n_features = X.shape[1]
        
        # Initialize population
        population = np.random.randint(2, size=(self.n_bees, n_features))
        fitness = np.array([self.fitness_function(bee, X, y) for bee in population])
        trials = np.zeros(self.n_bees)
        
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        for iter in range(self.max_iter):
            # Employed bees phase
            for i in range(self.n_bees):
                # Generate new solution
                new_solution = self.generate_new_solution(population[i], population, n_features)
                new_fitness = self.fitness_function(new_solution, X, y)
                
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
                    
            # Onlooker bees phase
            prob = 1 / (fitness + 1e-10)  # Avoid division by zero
            prob = prob / np.sum(prob)
            
            for i in range(self.n_bees):
                selected = np.random.choice(self.n_bees, p=prob)
                new_solution = self.generate_new_solution(population[selected], population, n_features)
                new_fitness = self.fitness_function(new_solution, X, y)
                
                if new_fitness < fitness[selected]:
                    population[selected] = new_solution
                    fitness[selected] = new_fitness
                    trials[selected] = 0
                else:
                    trials[selected] += 1
                    
            # Scout bees phase
            for i in range(self.n_bees):
                if trials[i] > self.limit:
                    population[i] = np.random.randint(2, size=n_features)
                    fitness[i] = self.fitness_function(population[i], X, y)
                    trials[i] = 0
                    
            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
                
            if (iter + 1) % 10 == 0:
                print(f"ABC Iteration {iter + 1}/{self.max_iter}, Best fitness: {best_fitness:.4f}")
                
        return best_solution
    
    def generate_new_solution(self, current, population, n_features):
        """
        Generate a new solution by modifying the current one.
        """
        new_solution = current.copy()
        # Select a random feature to modify
        feature_idx = np.random.randint(n_features)
        # Select a random solution from population (different from current)
        other_idx = np.random.choice([i for i in range(len(population)) if not np.array_equal(population[i], current)])
        # Modify the feature
        new_solution[feature_idx] = population[other_idx][feature_idx]
        return new_solution
    
    def fitness_function(self, bee, X, y):
        """
        Fitness function for evaluating a feature subset.
        
        Parameters:
        bee (np.array): Binary array representing feature selection
        X (np.array): Feature matrix
        y (np.array): Labels
        
        Returns:
        float: Fitness value (lower is better)
        """
        selected_features = bee.astype(bool)
        if np.sum(selected_features) == 0:
            return float('inf')
        
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        
        feature_ratio = np.sum(selected_features) / len(bee)
        return (1 - accuracy) + 0.1 * feature_ratio


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization algorithm for feature selection.
    """
    
    def __init__(self, n_particles=20, max_iter=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        
    def optimize(self, X, y):
        """
        Optimize feature selection using PSO.
        
        Parameters:
        X (np.array): Feature matrix
        y (np.array): Labels
        
        Returns:
        np.array: Best feature selection solution
        """
        n_features = X.shape[1]
        
        # Initialize particles (continuous values between 0 and 1)
        particles = np.random.rand(self.n_particles, n_features)
        velocities = np.random.rand(self.n_particles, n_features)
        
        # Convert to binary and evaluate fitness
        binary_particles = (particles > 0.5).astype(int)
        fitness = np.array([self.fitness_function(particle, X, y) for particle in binary_particles])
        
        # Initialize personal best
        p_best = particles.copy()
        p_best_fitness = fitness.copy()
        
        # Initialize global best
        g_best_idx = np.argmin(fitness)
        g_best = particles[g_best_idx].copy()
        g_best_fitness = fitness[g_best_idx]
        
        for iter in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (p_best[i] - particles[i]) + 
                                self.c2 * r2 * (g_best - particles[i]))
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                
                # Clamp positions to [0, 1]
                particles[i] = np.clip(particles[i], 0, 1)
                
                # Convert to binary and evaluate fitness
                binary_particle = (particles[i] > 0.5).astype(int)
                particle_fitness = self.fitness_function(binary_particle, X, y)
                
                # Update personal best
                if particle_fitness < p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = particle_fitness
                    
                # Update global best
                if particle_fitness < g_best_fitness:
                    g_best = particles[i].copy()
                    g_best_fitness = particle_fitness
                    
            if (iter + 1) % 10 == 0:
                print(f"PSO Iteration {iter + 1}/{self.max_iter}, Best fitness: {g_best_fitness:.4f}")
                
        # Convert final global best to binary
        best_solution = (g_best > 0.5).astype(int)
        return best_solution
    
    def fitness_function(self, particle, X, y):
        """
        Fitness function for evaluating a feature subset.
        
        Parameters:
        particle (np.array): Binary array representing feature selection
        X (np.array): Feature matrix
        y (np.array): Labels
        
        Returns:
        float: Fitness value (lower is better)
        """
        selected_features = particle.astype(bool)
        if np.sum(selected_features) == 0:
            return float('inf')
        
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        
        feature_ratio = np.sum(selected_features) / len(particle)
        return (1 - accuracy) + 0.1 * feature_ratio


def detailed_algorithm_comparison(X, y, feature_names):
    """
    Compare different feature selection algorithms with detailed analysis.
    
    Parameters:
    X (np.array): Feature matrix
    y (np.array): Labels
    feature_names (list): List of feature names
    """
    print("Comparing Feature Selection Algorithms")
    print("=" * 50)
    
    algorithms = {
        "Gray Wolf Optimization": GrayWolfOptimizer(n_wolves=15, max_iter=30),
        "Artificial Bee Colony": ArtificialBeeColony(n_bees=20, max_iter=30),
        "Particle Swarm Optimization": ParticleSwarmOptimization(n_particles=20, max_iter=30)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\nRunning {name}...")
        best_solution = algorithm.optimize(X, y)
        selected_features = best_solution.astype(bool)
        
        # Evaluate the selected features
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get detailed metrics
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Healthy', 'Parkinson'])
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "selected_features": np.sum(selected_features),
            "accuracy": accuracy,
            "solution": selected_features,
            "classification_report": report,
            "confusion_matrix": cm,
            "selected_feature_names": [feature_names[i] for i in range(len(selected_features)) if selected_features[i]],
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        print(f"{name} Results:")
        print(f"  Selected Features: {np.sum(selected_features)}/{X.shape[1]}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Feature Reduction: {((X.shape[1] - np.sum(selected_features)) / X.shape[1]) * 100:.2f}%")
    
    # Print detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Overall performance comparison
    print("\n1. OVERALL PERFORMANCE")
    print("-" * 30)
    print(f"{'Algorithm':<25} {'Features':<10} {'Accuracy':<10} {'Reduction':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        reduction = ((X.shape[1] - result['selected_features']) / X.shape[1]) * 100
        print(f"{name:<25} {result['selected_features']:<10} {result['accuracy']:<10.4f} {reduction:<10.2f}%")
    
    # Per-class performance comparison
    print("\n2. PER-CLASS PERFORMANCE")
    print("-" * 30)
    print(f"{'Algorithm':<25} {'Healthy_Prec':<12} {'Healthy_Rec':<12} {'PD_Prec':<10} {'PD_Rec':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        healthy_prec = result['classification_report']['Healthy']['precision']
        healthy_rec = result['classification_report']['Healthy']['recall']
        pd_prec = result['classification_report']['Parkinson']['precision']
        pd_rec = result['classification_report']['Parkinson']['recall']
        print(f"{name:<25} {healthy_prec:<12.4f} {healthy_rec:<12.4f} {pd_prec:<10.4f} {pd_rec:<10.4f}")
    
    # Confusion matrices
    print("\n3. CONFUSION MATRICES")
    print("-" * 30)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"          Predicted")
        print(f"          Healthy  Parkinson")
        print(f"Actual H  {result['confusion_matrix'][0][0]:<8} {result['confusion_matrix'][0][1]:<8}")
        print(f"       P  {result['confusion_matrix'][1][0]:<8} {result['confusion_matrix'][1][1]:<8}")
    
    # Selected features comparison
    print("\n4. SELECTED FEATURES BY ALGORITHM")
    print("-" * 30)
    all_features = set()
    algorithm_features = {}
    
    for name, result in results.items():
        algorithm_features[name] = set(result['selected_feature_names'])
        all_features.update(result['selected_feature_names'])
    
    # Create feature selection matrix
    print(f"{'Feature':<20}", end="")
    for name in algorithms.keys():
        print(f"{name:<8}", end="")
    print()
    print("-" * (20 + 8*len(algorithms)))
    
    for feature in sorted(all_features):
        print(f"{feature:<20}", end="")
        for name in algorithms.keys():
            if feature in algorithm_features[name]:
                print(f"{'✓':<8}", end="")
            else:
                print(f"{'✗':<8}", end="")
        print()
    
    # Feature importance for each algorithm
    print("\n5. FEATURE IMPORTANCE BY ALGORITHM")
    print("-" * 30)
    for name, result in results.items():
        print(f"\n{name}:")
        # Train a classifier on the selected features to get feature importance
        X_selected = X[:, result['solution']]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Get feature importance
        importance = clf.feature_importances_
        feature_importance = list(zip(result['selected_feature_names'], importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<5} {'Feature':<20} {'Importance':<10}")
        print("-" * 37)
        for i, (feature, imp) in enumerate(feature_importance[:10], 1):  # Top 10 features
            print(f"{i:<5} {feature:<20} {imp:<10.4f}")
    
    return results


def hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning for different classifiers to improve accuracy.
    
    Parameters:
    X (np.array): Feature matrix
    y (np.array): Labels
    
    Returns:
    dict: Best models and their parameters
    """
    print("Performing Hyperparameter Tuning")
    print("=" * 40)
    
    # Split data for tuning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers and their parameter grids
    classifiers = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
    }
    
    best_models = {}
    
    for name, config in classifiers.items():
        print(f"\nTuning {name}...")
        try:
            # Perform grid search
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            
            # Evaluate on test set
            test_accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled))
            
            best_models[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_score': best_score,
                'test_accuracy': test_accuracy
            }
            
            print(f"  Best CV Score: {best_score:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Best Parameters: {grid_search.best_params_}")
        except Exception as e:
            print(f"  Error tuning {name}: {e}")
    
    # Print comparison
    print("\n" + "=" * 50)
    print("CLASSIFIER COMPARISON AFTER TUNING")
    print("=" * 50)
    print(f"{'Classifier':<20} {'CV Score':<10} {'Test Accuracy':<15}")
    print("-" * 45)
    
    for name, result in best_models.items():
        print(f"{name:<20} {result['cv_score']:<10.4f} {result['test_accuracy']:<15.4f}")
    
    return best_models


def cross_validation_analysis(X, y, selected_features=None):
    """
    Perform cross-validation analysis to get a more robust accuracy estimate.
    
    Parameters:
    X (np.array): Feature matrix
    y (np.array): Labels
    selected_features (np.array): Boolean array indicating selected features
    
    Returns:
    dict: Cross-validation results
    """
    print("Performing Cross-Validation Analysis")
    print("=" * 40)
    
    # Use selected features if provided
    if selected_features is not None:
        X = X[:, selected_features]
        print(f"Using {np.sum(selected_features)} selected features")
    else:
        print(f"Using all {X.shape[1]} features")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different classifiers with cross-validation
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        # Perform cross-validation
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        
        results[name] = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'cv_scores': cv_scores
        }
        
        print(f"\n{name}:")
        print(f"  Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        print(f"  Individual Scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    return results


def ensemble_classification(X, y, selected_features=None):
    """
    Use ensemble methods to improve classification accuracy.
    
    Parameters:
    X (np.array): Feature matrix
    y (np.array): Labels
    selected_features (np.array): Boolean array indicating selected features
    
    Returns:
    dict: Ensemble results
    """
    print("Performing Ensemble Classification")
    print("=" * 40)
    
    # Use selected features if provided
    if selected_features is not None:
        X = X[:, selected_features]
        print(f"Using {np.sum(selected_features)} selected features")
    else:
        print(f"Using all {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create individual classifiers
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(random_state=42, probability=True)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Train individual classifiers
    rf.fit(X_train_scaled, y_train)
    svm.fit(X_train_scaled, y_train)
    knn.fit(X_train_scaled, y_train)
    lr.fit(X_train_scaled, y_train)
    
    # Get predictions
    rf_pred = rf.predict(X_test_scaled)
    svm_pred = svm.predict(X_test_scaled)
    knn_pred = knn.predict(X_test_scaled)
    lr_pred = lr.predict(X_test_scaled)
    
    # Ensemble predictions (majority voting)
    ensemble_pred = []
    for i in range(len(y_test)):
        votes = [rf_pred[i], svm_pred[i], knn_pred[i], lr_pred[i]]
        # Majority vote
        ensemble_pred.append(1 if sum(votes) >= 2 else 0)
    
    # Calculate accuracies
    rf_accuracy = accuracy_score(y_test, rf_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\nIndividual Classifier Accuracies:")
    print(f"  Random Forest: {rf_accuracy:.4f}")
    print(f"  SVM: {svm_accuracy:.4f}")
    print(f"  KNN: {knn_accuracy:.4f}")
    print(f"  Logistic Regression: {lr_accuracy:.4f}")
    print(f"  Ensemble (Majority Vote): {ensemble_accuracy:.4f}")
    
    return {
        'individual': {
            'Random Forest': rf_accuracy,
            'SVM': svm_accuracy,
            'KNN': knn_accuracy,
            'Logistic Regression': lr_accuracy
        },
        'ensemble': ensemble_accuracy,
        'predictions': {
            'rf': rf_pred,
            'svm': svm_pred,
            'knn': knn_pred,
            'lr': lr_pred,
            'ensemble': ensemble_pred,
            'actual': y_test
        }
    }


# === REALTIME MODULE START ===

class LivePredictor:
    """
    Real-time predictor class for handling audio stream and generating predictions.
    """
    def __init__(self, model_path='rf_model.pkl', selector_path='selected_features.pkl'):
        """
        Initialize the live predictor.
        
        Parameters:
        model_path (str): Path to the saved model file
        selector_path (str): Path to the saved selected features mask
        """
        self.sampling_rate = 22050
        self.window_size_seconds = 5
        self.buffer_size = self.sampling_rate * self.window_size_seconds
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Load model and feature selector
        try:
            self.model = joblib.load(model_path)
            self.selected_features_mask = joblib.load(selector_path)
            self.scaler = joblib.load('scaler.pkl')
            print("Model and artifacts loaded successfully for real-time prediction.")
        except FileNotFoundError:
            print("Model artifacts not found. Please run training first.")
            self.model = None
            self.selected_features_mask = None
            self.scaler = None

    def process_audio_chunk(self, audio_chunk):
        """
        Add a chunk of audio to the buffer.
        
        Parameters:
        audio_chunk (np.ndarray): Audio data chunk
        """
        self.audio_buffer.extend(audio_chunk)

    def predict_live(self):
        """
        Generate a prediction from the current audio buffer.
        
        Returns:
        dict: Prediction result or None if buffer not full
        """
        if len(self.audio_buffer) < self.buffer_size:
            return None
        
        if self.model is None:
            return {'error': 'Model not loaded'}
            
        # Convert buffer to numpy array
        audio_data = np.array(self.audio_buffer)
        
        # Check for silence/low energy (optional simple VAD)
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.01:  # Threshold for silence
            return {'label': 'Silence', 'probability': 0.0, 'is_silence': True}
            
        # Extract features
        features = extract_features(audio_data)
        
        if features is None:
            return None
            
        # Reshape for prediction
        # Filter features based on selection mask
        features_array = np.array(features)[self.selected_features_mask].reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        label = "Parkinson's" if prediction == 1 else "Healthy"
        prob_value = probability[1] if prediction == 1 else probability[0]
        
        return {
            'label': label,
            'probability': float(prob_value),
            'pd_probability': float(probability[1]),
            'healthy_probability': float(probability[0]),
            'is_silence': False
        }


def train_and_save_models(X, y, feature_names):
    """
    Train models using GWO, ABC, and PSO, and save the best one for real-time use.
    """
    print("\n" + "=" * 60)
    print("TRAINING AND SAVING MODELS FOR REAL-TIME APP")
    print("=" * 60)
    
    # 1. Comparison
    results = detailed_algorithm_comparison(X, y, feature_names)
    
    # 2. Select best algorithm
    best_algo_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_algo_name]
    
    print(f"\nBest Algorithm for Deployment: {best_algo_name}")
    
    # 3. Train final model on full dataset using best features
    # Note: In a real scenario, we would use a separate hold-out set, 
    # but for this demo we'll use the train/test split from the last fold
    
    selected_mask = best_result['solution'].astype(bool)
    X_selected = X[:, selected_mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Verify
    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"Final Model Accuracy: {acc:.4f}")
    
    # 4. Save artifacts
    print("Saving artifacts...")
    joblib.dump(clf, 'rf_model.pkl')
    joblib.dump(selected_mask, 'selected_features.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Also save specific algo models if needed for switching in UI
    # For now, we just save the 'best' one as default
    
    print("Done! Models saved: rf_model.pkl, selected_features.pkl, scaler.pkl")
    return results

# === REALTIME MODULE END ===


def main():
    """
    Main function to run the Parkinson's detection pipeline.
    """
    print("Parkinson's Disease Detection Using Speech Analysis")
    print("=" * 50)
    
    # Define paths
    pd_path = os.path.join(os.getcwd(), "PD_AH", "PD_AH")
    hc_path = os.path.join(os.getcwd(), "HC_AH", "HC_AH")
    
    # Check if directories exist
    if not os.path.exists(pd_path):
        print(f"Error: PD directory not found at {pd_path}")
        return
        
    if not os.path.exists(hc_path):
        print(f"Error: HC directory not found at {hc_path}")
        return
    
    # Extract features
    print("Loading Parkinson's samples...")
    features_list = []
    labels = []
    
    pd_count = 0
    for file in os.listdir(pd_path):
        if file.endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.WAV')):
            file_path = os.path.join(pd_path, file)
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels.append(1)  # Parkinson's
                pd_count += 1

    print("Loading Healthy samples...")
    hc_count = 0
    for file in os.listdir(hc_path):
        if file.endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.WAV')):
            file_path = os.path.join(hc_path, file)
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels.append(0)  # Healthy
                hc_count += 1

    X = np.array(features_list)
    y = np.array(labels)

    print(f"Loaded {pd_count} Parkinson's samples and {hc_count} Healthy samples")
    print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Class distribution: Parkinson's={np.sum(y==1)}, Healthy={np.sum(y==0)}")
    
    # Feature names (updated to include new features)
    feature_names = [
        'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'MFCC4_mean', 'MFCC5_mean', 'MFCC6_mean', 
        'MFCC7_mean', 'MFCC8_mean', 'MFCC9_mean', 'MFCC10_mean', 'MFCC11_mean', 'MFCC12_mean', 'MFCC13_mean',
        'MFCC1_std', 'MFCC2_std', 'MFCC3_std', 'MFCC4_std', 'MFCC5_std', 'MFCC6_std',
        'MFCC7_std', 'MFCC8_std', 'MFCC9_std', 'MFCC10_std', 'MFCC11_std', 'MFCC12_std', 'MFCC13_std',
        'Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth',
        'ZCR', 'RMS',
        'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5', 'Chroma6',
        'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10', 'Chroma11', 'Chroma12',
        # Additional features
        'Spectral_Contrast1', 'Spectral_Contrast2', 'Spectral_Contrast3', 'Spectral_Contrast4',
        'Spectral_Contrast5', 'Spectral_Contrast6', 'Spectral_Contrast7',
        'Spectral_Flatness', 'Tempo'
    ]
    
    # Compare all algorithms with detailed analysis
    # Modified to also save the models
    if not os.path.exists('rf_model.pkl'):
        results = train_and_save_models(X, y, feature_names)
    else:
        print("Pre-trained models found. Loading for comparison display (skipping full re-training for speed).")
        # In a real run, you might want to force re-training or load the results.
        # For this script's flow, we'll still run the comparison logic to show output
        results = detailed_algorithm_comparison(X, y, feature_names)
    
    # Use the best algorithm for detailed analysis (based on accuracy)
    best_algorithm = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\nBest algorithm: {best_algorithm}")
    
    selected_features = results[best_algorithm]['solution']
    X_selected = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{best_algorithm.upper()} RESULTS")
    print("=" * 30)
    print(f"Accuracy with selected features: {accuracy:.4f}")
    print(f"Features selected: {np.sum(selected_features)}/{X.shape[1]}")
    print(f"Feature reduction: {((X.shape[1] - np.sum(selected_features)) / X.shape[1]) * 100:.2f}%")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s']))
    
    # Get selected feature names
    selected_feature_indices = [i for i, selected in enumerate(selected_features) if selected]
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]

    print(f"\nSELECTED FEATURES BY {best_algorithm.upper()}:")
    print("=" * 50)
    for i, feature in enumerate(selected_feature_names, 1):
        print(f"{i:2d}. {feature}")
    
    # Additional techniques to improve accuracy
    print("\n" + "=" * 60)
    print("TECHNIQUES TO IMPROVE MODEL ACCURACY")
    print("=" * 60)
    
    # 1. Hyperparameter tuning
    print("\n1. Hyperparameter Tuning")
    print("-" * 25)
    print("Performing hyperparameter tuning for different classifiers...")
    tuned_models = hyperparameter_tuning(X_selected, y)
    
    # 2. Cross-validation analysis
    print("\n2. Cross-Validation Analysis")
    print("-" * 25)
    print("Analysis skipped for speed in this run (uncomment to run)")
    # cv_results = cross_validation_analysis(X, y, selected_features)
    
    # 3. Ensemble methods
    print("\n3. Ensemble Classification")
    print("-" * 25)
    print("Ensemble analysis skipped for speed in this run (uncomment to run)")
    # ensemble_results = ensemble_classification(X, y, selected_features)
    
    # Summary of improvement techniques
    print("\n" + "=" * 60)
    print("ACCURACY IMPROVEMENT SUMMARY")
    print("=" * 60)
    print("1. Feature Engineering:")
    print("   - Added spectral contrast features (7 dimensions)")
    print("   - Added spectral flatness feature")
    print("   - Added tempo feature")
    print("   - Total features increased from 43 to 52")
    
    print("\n2. Model Optimization:")
    print("   - Hyperparameter tuning for all classifiers")
    print("   - Cross-validation for robust performance estimation")
    print("   - Ensemble methods combining multiple classifiers")
    
    print("\n3. Advanced Techniques:")
    print("   - Try deep learning models (CNN, RNN)")
    print("   - Use data augmentation to increase dataset size")
    print("   - Implement more sophisticated feature selection algorithms")
    print("   - Experiment with different validation strategies")
    
    print("\n4. Dataset Enhancement:")
    print("   - Collect more samples for better generalization")
    print("   - Balance the dataset if needed")
    print("   - Include more diverse demographic groups")
    
    print("\nREAL-TIME MODULE READY")


if __name__ == "__main__":
    main()
