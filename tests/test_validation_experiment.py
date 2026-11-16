import json
from pathlib import Path
from agents.adaptation.utils import collect_run_metrics

def test_validation_metrics_ignore_test_stdout_and_other_models(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENTCITY_VALIDATION_ONLY','1')
    for model, score in [('wanted',4.2),('other',0.1)]:
        p=tmp_path/'libcity/cache'/model/'training_evidence.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'model':model,'dataset':'D','best_validation_mae':score,'history':[{}]*35,'device':'mps'}))
    metrics, detail=collect_run_metrics(tmp_path,model_name='wanted',dataset='D',stdout='masked_mae: 0.001')
    assert metrics=={'masked_mae':4.2}
    assert detail['partition']=='validation'
    assert detail['completed_epochs']==35

def test_missing_validation_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENTCITY_VALIDATION_ONLY','1')
    metrics, detail=collect_run_metrics(tmp_path,model_name='missing',dataset='D',stdout='masked_mae: 0.001')
    assert metrics=={}
    assert detail['source_kind']=='validation_missing'
