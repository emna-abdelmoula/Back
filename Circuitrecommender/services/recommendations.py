import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Circuitrecommender.models import Tourism

def recommend_destinations(user_preferences, num_recommendations=5):
    # Charger les données de la base de données dans un DataFrame
    queryset = Tourism.objects.all().values()
    df = pd.DataFrame(queryset)
    
    # # Remplir les valeurs NaN
    # df["Dietaryrestrictions"] = df["Dietaryrestrictions"].fillna("free")
    # df["price"] = df["price"].fillna("")
    # df["cuisine"] = df["cuisine"].fillna("")
    
    # Créer la colonne destinations_features
    df["destinations_features"] = df["subcategory_name"] + ' ' + df["price"]
    
    # Initialiser CountVectorizer
    vectorizer = CountVectorizer()
    
    # Créer la matrice de caractéristiques pour les destinations
    destinations_features = vectorizer.fit_transform(df["destinations_features"])
    
    # Créer le vecteur utilisateur
    user_vector = vectorizer.transform([user_preferences])
    
    # Calculer la similarité cosinus
    cosine_sim = cosine_similarity(user_vector, destinations_features)
    
    # Trier les scores de similarité
    scores = [(idx, sim) for idx, sim in enumerate(cosine_sim[0])]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Recommander les destinations
    recommendations = []
    for i in scores[:num_recommendations]:
        index = i[0]
        destinations_details = {
            "recommended_destinations_ids": str(df.iloc[index]['id']),
            "category_name": df.iloc[index]['category_name'],
            "subcategory_name": df.iloc[index]['subcategory_name'],
            "subsubcategory_name": df.iloc[index]['subsubcategory'],
            "Photo": df.iloc[index]['url'],
            "name": df.iloc[index]['name'],
            # "details": df.iloc[index]['details'],
            "Price": df.iloc[index]['price']
        }
        recommendations.append(destinations_details)
    
    return recommendations