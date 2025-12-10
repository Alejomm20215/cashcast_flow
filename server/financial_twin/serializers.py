from __future__ import annotations

from rest_framework import serializers

from .models import SimulationRun


class SimulationRequestSerializer(serializers.Serializer):
    name = serializers.CharField(required=False, allow_blank=True, max_length=120)
    ledger = serializers.FileField()
    months = serializers.IntegerField(min_value=1, default=12)
    annual_inflation = serializers.FloatField(min_value=0.0, default=0.02)
    engine = serializers.ChoiceField(choices=["naive", "prophet"], default="naive")
    paths = serializers.IntegerField(min_value=100, max_value=100000, default=5000)
    return_mean = serializers.FloatField(default=0.07)
    return_vol = serializers.FloatField(default=0.15)
    inflation_mean = serializers.FloatField(default=0.03)
    inflation_std = serializers.FloatField(default=0.01)
    shock_lambda = serializers.FloatField(default=0.1)
    shock_mean = serializers.FloatField(default=200.0)
    start_wealth = serializers.FloatField(default=0.0)
    goal_target = serializers.FloatField(required=False, allow_null=True)
    liquidity_floor = serializers.FloatField(required=False, allow_null=True)
    seed = serializers.IntegerField(required=False)


class ScenarioRequestSerializer(SimulationRequestSerializer):
    delta = serializers.FileField()


class SimulationRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationRun
        fields = [
            "id",
            "created_at",
            "name",
            "kind",
            "status",
            "ledger_path",
            "delta_path",
            "config",
            "result_path",
            "error",
        ]

