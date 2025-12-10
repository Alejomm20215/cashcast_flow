from django.urls import include, path

urlpatterns = [
    path("api/", include("financial_twin.urls")),
]

