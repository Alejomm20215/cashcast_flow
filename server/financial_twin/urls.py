from django.urls import path

from .views import RunDetailView, ScenarioView, SimulationView

urlpatterns = [
    path("simulations/", SimulationView.as_view(), name="simulation-create"),
    path("scenarios/", ScenarioView.as_view(), name="scenario-create"),
    path("runs/<int:pk>/", RunDetailView.as_view(), name="run-detail"),
]

