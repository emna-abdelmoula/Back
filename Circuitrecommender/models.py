from django.db import models
from django.conf import settings

    
class Tourism(models.Model):
    """
    Django Tourism Model
    """
    # ID (optional if you want to include it as an auto-increment field, Django adds it by default)
    # id = models.AutoField(primary_key=True)

    # Category name (e.g., culinary, Historical and Arts, Nature and Adventure)
    category_name = models.CharField(max_length=64, null=False)

    # Subcategory name (e.g., Cafe, Museums, Nature)
    subcategory_name = models.CharField(max_length=64, null=False)

    # Subsubcategory name (e.g., Inland Coffee, History Museums, Beach)
    subsubcategory = models.CharField(max_length=64, null=False)

    # Rating (e.g., 4.5, 5)
    rating = models.FloatField(null=True, blank=True)

    # URL (e.g., URL to the image)
    url = models.URLField(max_length=1024, null=True, blank=True)

    # Name (e.g., Cafe Ben Yedder)
    name = models.CharField(max_length=256, null=False)

    # Address (e.g., Flughafen, Tunis Tunisia)
    address = models.CharField(max_length=512, null=True, blank=True)

    # Latitude (optional, if present)
    latitude = models.FloatField(null=True, blank=True)

    # Longitude (optional, if present)
    longitude = models.FloatField(null=True, blank=True)
    
    cuisine = models.CharField(max_length=255, null = True )
    
    Dietaryrestrictions = models.CharField(max_length=1024, null=True)
    
    Meals = models.CharField(max_length= 255, null = True) 
    
    price=models.CharField(max_length=512, null=True)
    
    Dishes=models.CharField(max_length=512, null=True)
    
    GoodFor = models.CharField(max_length=255, null = True )
    
    Duration = models.CharField(max_length=1024, null=True)
    destinations_features= models.CharField(max_length=1024, null=True)
    

    