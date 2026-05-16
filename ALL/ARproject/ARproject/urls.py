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
    path('missions/', views.missions, name='missions'),
    path('achievements/', views.achievements, name='achievements'),
    path('equipment/equip/', views.equipment_equip, name='equipment_equip'),
    path('equipment/', views.equipment, name='equipment'),
    path('settings/', views.settings_page, name='settings'),
    path('api/realm/save/', views.realm_save_api, name='realm_save_api'),
    path('api/center/start/', views.center_start_api, name='center_start_api'),
    path('api/center/stop/', views.center_stop_api, name='center_stop_api'),
    path('api/center/status/', views.center_status_api, name='center_status_api'),
    path('api/center/resume/', views.center_resume_api, name='center_resume_api'),
    path('api/center/reset-reconstruction/', views.center_reset_reconstruction_api, name='center_reset_reconstruction_api'),
    path('api/center/latest-mesh/', views.center_latest_mesh_api, name='center_latest_mesh_api'),
    path('api/reality-override/save/', views.reality_override_save_api, name='reality_override_save_api'),
    path('api/depth-scan/import/', views.depth_scan_import_api, name='depth_scan_import_api'),
    path('api/reality-scan/clear/', views.reality_scan_clear_api, name='reality_scan_clear_api'),
    path('depth-scan/tester/', views.depth_scan_tester, name='depth_scan_tester'),
    path('api/progress/event/', views.progress_event_api, name='progress_event_api'),
    path('realm/editor/', views.realm_editor, name='realm_editor'),
    path('realm/', views.realm, name='realm'),
    path('training/', views.training, name='training'),
    path('reality-override/', views.reality_override, name='reality_override'),
    path('my-realm/', views.my_realm, name='my_realm'),
    path('privacy/', views.privacy_policy, name='privacy'),
    path('terms/', views.terms_of_service, name='terms'),
    path('security/', views.security_center, name='security'),
    path('disclaimer/', views.disclaimer, name='disclaimer'),
    path('devices/', views.devices, name='devices'),
    path('material-library/', views.material_library, name='material_library'),
    path('worlds/', views.worlds_plaza, name='worlds'),
    path('progress/', views.progress_hub, name='progress_hub'),
    path('guide/', views.guide_hub, name='guide'),
    path('tutorial/', views.tutorial, name='tutorial'),
    path('about/', views.about, name='about'),
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])