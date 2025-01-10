import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, f_oneway
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
def analyze_data_quality(df):
    """
    Analyse la qualité des données d'un DataFrame sans suppositions sur les colonnes.
    """
    print("=" * 80)
    print("RAPPORT D'ANALYSE DE LA QUALITÉ DES DONNÉES")
    print("=" * 80)

    # 1. Analyse des valeurs manquantes
    print("\n1. ANALYSE DES VALEURS MANQUANTES")
    print("-" * 40)
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100

    if missing_values.sum() > 0:
        print("\nColonnes avec valeurs manquantes:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"{col}: {missing} valeurs manquantes ({missing_percentages[col]:.2f}%)")
    else:
        print("Aucune valeur manquante trouvée dans le dataset")

    # 2. Analyse des doublons
    print("\n2. ANALYSE DES DOUBLONS")
    print("-" * 40)
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    print(f"Nombre total de lignes dupliquées: {len(duplicate_rows)}")

    # 3. Analyse des valeurs aberrantes (pour colonnes numériques)
    print("\n3. ANALYSE DES VALEURS ABERRANTES")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        if len(outliers) > 0:
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'min': outliers.min(),
                'max': outliers.max()
            }

    if outliers_summary:
        print("\nValeurs aberrantes détectées:")
        for col, stats in outliers_summary.items():
            print(f"\n{col}:")
            print(f"  - Nombre de valeurs aberrantes: {stats['count']}")
            print(f"  - Pourcentage: {stats['percentage']:.2f}%")
            print(f"  - Plage des valeurs aberrantes: [{stats['min']:.2f}, {stats['max']:.2f}]")

    # 4. Analyse des dates (si présentes)
    print("\n4. ANALYSE DES DATES")
    print("-" * 40)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    
    if len(date_cols) > 0:
        for col in date_cols:
            print(f"\nAnalyse de la colonne {col}:")
            print(f"  - Période: de {df[col].min()} à {df[col].max()}")
            invalid_dates = df[df[col].isnull()]
            if len(invalid_dates) > 0:
                print(f"  - Dates invalides: {len(invalid_dates)} lignes")

    # 5. Visualisation des distributions
    if len(numeric_cols) > 0:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) - 1) // n_cols + 1
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution de {col}')
            plt.xticks(rotation=45)
        plt.tight_layout()

    # 6. Analyse des variables catégorielles
    print("\n5. ANALYSE DES VARIABLES CATÉGORIELLES")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        unique_values = df[col].nunique()
        value_counts = df[col].value_counts()
        print(f"\n{col}:")
        print(f"  - Nombre de valeurs uniques: {unique_values}")
        if unique_values < 10:  # Afficher la distribution seulement si peu de valeurs uniques
            print("  - Distribution:")
            for val, count in value_counts.items():
                print(f"    {val}: {count} ({count/len(df)*100:.2f}%)")

    # 7. Vérification des formats
    print("\n6. VÉRIFICATION DES FORMATS")
    print("-" * 40)
    print("\nTypes de données par colonne:")
    print(df.dtypes)

    return {
        'missing_values': missing_values,
        'duplicate_count': len(duplicate_rows),
        'outliers': outliers_summary
    }

def analyze_relationships(df, target_column=None):
    """
    Analyse les relations entre les variables du DataFrame.
    Si target_column est spécifié, analyse les relations par rapport à cette variable cible.
    """
    if target_column and target_column not in df.columns:
        raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le DataFrame")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    results = {
        'numerical_analysis': {},
        'categorical_analysis': {}
    }

    # Analyse des relations numériques
    if len(numeric_cols) > 0:
        # Matrice de corrélation
        correlation_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matrice de corrélation des variables numériques')
        plt.tight_layout()

        # Analyse par rapport à la variable cible si spécifiée
        if target_column and df[target_column].nunique() == 2:  # Pour analyse binaire
            for col in numeric_cols:
                if col != target_column:
                    group1 = df[df[target_column] == df[target_column].unique()[0]][col]
                    group2 = df[df[target_column] == df[target_column].unique()[1]][col]
                    stat, p_value = stats.mannwhitneyu(group1, group2)
                    results['numerical_analysis'][col] = {
                        'statistic': stat,
                        'p_value': p_value
                    }

    # Analyse des relations catégorielles
    if target_column:
        for col in categorical_cols:
            if col != target_column:
                contingency_table = pd.crosstab(df[col], df[target_column])
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                results['categorical_analysis'][col] = {
                    'chi2': chi2,
                    'p_value': p_value
                }

    return results

def prepare_data(df):
    """
    Prépare les données en convertissant les colonnes de dates et en créant des caractéristiques temporelles.
    """
    df_prepared = df.copy()
    
    # Identification et conversion des colonnes de dates
    for col in df.columns:
        try:
            df_prepared[col] = pd.to_datetime(df[col])
            print(f"Colonne convertie en datetime: {col}")
            
            # Création des caractéristiques temporelles
            df_prepared[f'{col}_year'] = df_prepared[col].dt.year
            df_prepared[f'{col}_month'] = df_prepared[col].dt.month
            df_prepared[f'{col}_day'] = df_prepared[col].dt.day
            
        except (ValueError, TypeError):
            continue
    
    return df_prepared

def main_analysis(data, target_column=None):
    """
    Effectue l'analyse principale des données.
    
    Parameters:
    data (pandas.DataFrame): Le DataFrame à analyser
    target_column (str, optional): La colonne cible pour l'analyse des relations
    
    Returns:
    dict: Un dictionnaire contenant les résultats de l'analyse
    """
    print("DÉBUT DE L'ANALYSE\n")
    
    # Préparation des données
    df = prepare_data(data)
    
    # Analyse de la qualité des données
    print("\nANALYSE DE LA QUALITÉ DES DONNÉES")
    quality_results = analyze_data_quality(df)
    
    # Analyse des relations
    if target_column:
        print(f"\nANALYSE DES RELATIONS AVEC LA VARIABLE CIBLE: {target_column}")
        relationship_results = analyze_relationships(df, target_column)
        
        # Affichage des relations significatives
        significance_threshold = 0.05
        
        print("\nVariables numériques significativement associées à la variable cible (p < 0.05):")
        for var, stats in relationship_results['numerical_analysis'].items():
            if stats['p_value'] < significance_threshold:
                print(f"- {var}: p-value = {stats['p_value']:.4f}")
        
        print("\nVariables catégorielles significativement associées à la variable cible (p < 0.05):")
        for var, stats in relationship_results['categorical_analysis'].items():
            if stats['p_value'] < significance_threshold:
                print(f"- {var}: p-value = {stats['p_value']:.4f}")
    
    return {
        'quality_analysis': quality_results,
        'relationship_analysis': relationship_results if target_column else None
    }
    
    
    


def create_visualizations(df, target_column):
    """
    Crée une série de visualisations détaillées avec Seaborn.
    """
    print("\nCRÉATION DES VISUALISATIONS")
    print("=" * 80)
    
    # Définition des paramètres de visualisation
    N_COLS = 3  # Nombre de colonnes pour les subplots
    
    target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Configuration du style Seaborn
    print("Configuration du style Seaborn...")
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 1. Distribution de la variable cible
    plt.figure(figsize=(12, 6))
    if target_is_numeric:
        sns.histplot(data=df, x=target_column, kde=True)
        plt.title(f"Distribution de {target_column}")
    else:
        sns.countplot(data=df, x=target_column)
        plt.title(f"Répartition de {target_column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 2. Pair plots pour variables numériques
    numeric_vars = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    if len(numeric_vars) >= 2:
        print("\nCréation du pair plot (peut prendre un moment)...")
        sns.pairplot(df[list(numeric_vars) + [target_column]], 
                    hue=target_column if not target_is_numeric else None,
                    diag_kind='kde')
        plt.tight_layout()
        
    
    """  # 3. Box plots pour variables numériques
    if len(numeric_cols) > 0:
        n_rows = (len(numeric_cols) - 1) // N_COLS + 1
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(15, 5*n_rows))
        if n_rows * N_COLS > 1:
            axes = axes.ravel()
        else:
            axes = [axes]
        
        for idx, col in enumerate(numeric_cols):
            if col != target_column and idx < len(axes):
                if not target_is_numeric:
                    sns.boxplot(data=df, x=target_column, y=col, ax=axes[idx])
                else:
                    sns.boxplot(data=df, y=col, ax=axes[idx])
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
                axes[idx].set_title(f"Distribution de {col}")
        
        # Masquer les axes vides
        for idx in range(len(numeric_cols), len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
                
        plt.tight_layout()
    
    # 4. Violin plots
    if len(numeric_cols) > 0 and not target_is_numeric:
        n_rows = (len(numeric_cols) - 1) // N_COLS + 1
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(15, 5*n_rows))
        if n_rows * N_COLS > 1:
            axes = axes.ravel()
        else:
            axes = [axes]
        
        for idx, col in enumerate(numeric_cols):
            if col != target_column and idx < len(axes):
                sns.violinplot(data=df, x=target_column, y=col, ax=axes[idx])
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
                axes[idx].set_title(f"Distribution de {col} par {target_column}")
        
        # Masquer les axes vides
        for idx in range(len(numeric_cols), len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
                
        plt.tight_layout()
    
    # 5. Heatmap des corrélations
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                   cmap='coolwarm', center=0, fmt='.2f',
                   linewidths=0.5)
        plt.title("Matrice de corrélation des variables numériques")
        plt.tight_layout()
    
    # 6. Catplots pour variables catégorielles
    if len(categorical_cols) > 0:
        n_rows = (len(categorical_cols) - 1) // N_COLS + 1
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(15, 5*n_rows))
        if n_rows * N_COLS > 1:
            axes = axes.ravel()
        else:
            axes = [axes]
        
        for idx, col in enumerate(categorical_cols):
            if col != target_column and idx < len(axes):
                if target_is_numeric:
                    sns.barplot(data=df, x=col, y=target_column, ax=axes[idx])
                else:
                    sns.countplot(data=df, x=col, hue=target_column, ax=axes[idx])
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
                axes[idx].set_title(f"{col} vs {target_column}")
        
        # Masquer les axes vides
        for idx in range(len(categorical_cols), len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
                
        plt.tight_layout() """
    
    
def analyze_categorical_variable(df, col, target_column, target_is_numeric):
    """
    Analyse une variable catégorielle avec gestion des erreurs.
    """
    try:
        if target_is_numeric:
            # Vérifier qu'il y a au moins 2 groupes non vides
            groups = [group[target_column].values for name, group in df.groupby(col) 
                     if len(group[target_column].values) > 0]
            
            if len(groups) < 2:
                return {
                    'test': "ANOVA impossible",
                    'statistic': None,
                    'p_value': 1.0,
                    'error': "Pas assez de groupes pour ANOVA"
                }
            
            statistic, p_value = f_oneway(*groups)
            test_name = "ANOVA"
        else:
            # Pour le test du Chi2, vérifier que le tableau de contingence n'est pas vide
            contingency = pd.crosstab(df[col], df[target_column])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return {
                    'test': "Chi2 impossible",
                    'statistic': None,
                    'p_value': 1.0,
                    'error': "Pas assez de catégories pour Chi2"
                }
            
            statistic, p_value, dof, expected = chi2_contingency(contingency)
            test_name = "Chi2"

        return {
            'test': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'error': None
        }
    except Exception as e:
        return {
            'test': "Test impossible",
            'statistic': None,
            'p_value': 1.0,
            'error': str(e)
        }

def analyze_numeric_variable(df, col, target_column, target_is_numeric, target_is_binary):
    """
    Analyse une variable numérique avec gestion des erreurs.
    """
    try:
        if target_is_numeric:
            # Test de Spearman
            valid_data = df[[col, target_column]].dropna()
            if len(valid_data) < 2:
                return {
                    'test': "Spearman impossible",
                    'statistic': None,
                    'p_value': 1.0,
                    'error': "Pas assez de données valides"
                }
            
            correlation, p_value = spearmanr(valid_data[col], valid_data[target_column])
            test_name = "Spearman"
            
        elif target_is_binary:
            # Test de Mann-Whitney
            groups = [group[col].dropna().values for name, group in df.groupby(target_column)]
            if any(len(group) < 1 for group in groups):
                return {
                    'test': "Mann-Whitney impossible",
                    'statistic': None,
                    'p_value': 1.0,
                    'error': "Pas assez de données dans les groupes"
                }
            
            statistic, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            correlation = statistic / (len(groups[0]) * len(groups[1]))
            test_name = "Mann-Whitney U"
            
        else:
            # ANOVA
            groups = [group[col].dropna().values for name, group in df.groupby(target_column) 
                     if len(group[col].dropna().values) > 0]
            if len(groups) < 2:
                return {
                    'test': "ANOVA impossible",
                    'statistic': None,
                    'p_value': 1.0,
                    'error': "Pas assez de groupes pour ANOVA"
                }
            
            statistic, p_value = f_oneway(*groups)
            correlation = statistic
            test_name = "ANOVA"

        return {
            'test': test_name,
            'statistic': correlation,
            'p_value': p_value,
            'error': None
        }
    except Exception as e:
        return {
            'test': "Test impossible",
            'statistic': None,
            'p_value': 1.0,
            'error': str(e)
        }


def analyze_target_relationships(df, target_column):
    """
    Analyse détaillée des relations entre la variable cible et les autres variables.
    """
    results = {
        'dependent_vars': [],
        'independent_vars': [],
        'correlation_scores': {},
        'statistical_tests': {},
        'errors': {}
    }

    target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    target_is_binary = len(df[target_column].dropna().unique()) == 2

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        if col != target_column:
            test_results = analyze_numeric_variable(df, col, target_column, target_is_numeric, target_is_binary)
            results['statistical_tests'][col] = test_results
            if test_results['error']:
                results['errors'][col] = test_results['error']
            elif test_results['p_value'] < 0.05:
                results['dependent_vars'].append(col)
            else:
                results['independent_vars'].append(col)

    for col in categorical_cols:
        if col != target_column:
            test_results = analyze_categorical_variable(df, col, target_column, target_is_numeric)
            results['statistical_tests'][col] = test_results
            if test_results['error']:
                results['errors'][col] = test_results['error']
            elif test_results['p_value'] < 0.05:
                results['dependent_vars'].append(col)
            else:
                results['independent_vars'].append(col)

    # Vérification des variables numériques avant le calcul des scores
    if len(numeric_cols) > 0:
        numeric_features = df[numeric_cols].drop(target_column, axis=1, errors='ignore')
        if not numeric_features.empty:
            try:
                if target_is_numeric:
                    mi_scores = mutual_info_regression(numeric_features, df[target_column])
                else:
                    mi_scores = mutual_info_classif(numeric_features, df[target_column])
                for feat, mi_score in zip(numeric_features.columns, mi_scores):
                    results['correlation_scores'][feat] = mi_score
            except Exception as e:
                print(f"Erreur lors du calcul des scores d'information mutuelle : {e}")

    # Vérification des variables dépendantes avant la création de visualisations
    if results['dependent_vars']:
        try:
            create_visualizations(df, target_column)
        except Exception as e:
            print(f"Erreur lors de la création des visualisations : {e}")
    else:
        print("Aucune variable dépendante trouvée, les visualisations ne seront pas créées.")

    return results