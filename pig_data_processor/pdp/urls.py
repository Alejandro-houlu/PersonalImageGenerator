from os import name
from django.urls import path
from . import views

urlpatterns = [
    path("index", views.index, name="index"),
    path("ping", views.pong, name='pong'),
    path("userInfo", views.userInfo, name='userInfo'),
    path("userRecommendation", views.userRecommendation, name='userRecommendation'),
    path("recommendationByUser", views.recommendationByUser, name='recommendationByUser'),
    path("itemRecommendation", views.itemRecommendation, name='itemRecommendation'),
    path("recommendationByItem", views.recommendationByItem, name='recommendationByItem'),
    path("generateImage", views.generateImage, name='generateImage'),
    path("generatePrompt", views.generatePrompt, name='generatePrompt'),
    path("searchItem", views.searchItem, name='searchItem'),
    path("getPersonalizedItems", views.getPersonalizedItems, name='getPersonalizedItems'),
    path("uploadImage", views.uploadImage, name='uploadImage'),

]
