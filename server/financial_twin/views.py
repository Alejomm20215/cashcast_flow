import json
import uuid
from pathlib import Path

from django.conf import settings
from django.http import Http404
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import SimulationRun
from .serializers import (
    ScenarioRequestSerializer,
    SimulationRequestSerializer,
    SimulationRunSerializer,
)
from .tasks import run_simulation


def _save_uploaded(upload, folder: str) -> Path:
    filename = f"{uuid.uuid4()}_{upload.name}"
    path = Path(settings.MEDIA_ROOT) / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for chunk in upload.chunks():
            f.write(chunk)
    return path


class SimulationView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        serializer = SimulationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        ledger_file = data.pop("ledger")
        ledger_path = _save_uploaded(ledger_file, "ledgers")

        run = SimulationRun.objects.create(
            name=data.get("name", ""),
            kind="simulate",
            status="pending",
            ledger_path=str(ledger_path),
            config=data,
        )

        run_simulation.delay(run.id)
        return Response(SimulationRunSerializer(run).data, status=status.HTTP_202_ACCEPTED)


class ScenarioView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        serializer = ScenarioRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        ledger_file = data.pop("ledger")
        delta_file = data.pop("delta")
        ledger_path = _save_uploaded(ledger_file, "ledgers")
        delta_path = _save_uploaded(delta_file, "deltas")

        run = SimulationRun.objects.create(
            name=data.get("name", ""),
            kind="scenario",
            status="pending",
            ledger_path=str(ledger_path),
            delta_path=str(delta_path),
            config=data,
        )

        run_simulation.delay(run.id)
        return Response(SimulationRunSerializer(run).data, status=status.HTTP_202_ACCEPTED)


class RunDetailView(APIView):
    def get(self, request, pk: int):
        try:
            run = SimulationRun.objects.get(pk=pk)
        except SimulationRun.DoesNotExist as exc:
            raise Http404 from exc

        payload = SimulationRunSerializer(run).data
        if run.result_path and Path(run.result_path).exists():
            payload["result"] = json.loads(Path(run.result_path).read_text())
        else:
            payload["result"] = None
        return Response(payload)

