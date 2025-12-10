from django.db import models


class SimulationRun(models.Model):
    KIND_CHOICES = [
        ("simulate", "Simulate"),
        ("scenario", "Scenario"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=120, blank=True, default="")
    kind = models.CharField(max_length=32, choices=KIND_CHOICES, default="simulate")
    status = models.CharField(max_length=32, default="pending")
    ledger_path = models.CharField(max_length=255)
    delta_path = models.CharField(max_length=255, blank=True, default="")
    config = models.JSONField(default=dict)
    result_path = models.CharField(max_length=255, blank=True, default="")
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]

