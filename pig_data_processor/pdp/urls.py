from os import name
from django.urls import path
from . import views

urlpatterns = [
    path("index", views.index, name="index"),
    path("ping", views.pong, name='pong'),
    path("userInfo", views.userInfo, name='userInfo'),
    path("userRecommendation", views.userRecommendation, name='userRecommendation'),
    path("recommendationByUser", views.recommendationByUser, name='recommendationByUser')
]
