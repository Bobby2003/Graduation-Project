"""
URL configuration for ARproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from ARapp import views

from django.conf.urls import handler404
handler404 = 'ARapp.views.error_404'

urlpatterns = [
path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('admin/', admin.site.urls),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('market/', views.market, name='market'),
    path('ranking/', views.ranking, name='ranking'),
    path('scanner/', views.view_scanner, name='scanner'),  # 扫描器主页
    path('video_feed/', views.video_feed, name='video_feed'),  # 真实的视频流数据接口
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])