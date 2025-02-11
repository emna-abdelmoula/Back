"""
URL configuration for recommender project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from Circuitrecommender.views import  get_categories_view, get_prices_view, get_subcategories_view, recommend_destinations_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('recommend-destinations/', recommend_destinations_view, name='recommend_destinations_view'),
    path('api/v1/auth/',include('Users.urls')),
    path('api/get-categories/', get_categories_view, name='get_categories'),
    path('api/get-subcategories/', get_subcategories_view, name='get_subcategories'),
    path('api/get-prices/', get_prices_view, name='get_prices'),
]
