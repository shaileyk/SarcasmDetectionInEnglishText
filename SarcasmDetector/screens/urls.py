from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('next/<int:step>', views.next),
    path('training/', views.training, name='training'),
    path('performance/', views.performance, name='performance'),
    path('get_random/', views.get_random),
    path('get_tweet/', views.get_tweet),
]
