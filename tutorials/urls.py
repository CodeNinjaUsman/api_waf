from django.urls import path
from tutorials import views 
 
urlpatterns = [ 
    path('', views.tutorial_list)
]