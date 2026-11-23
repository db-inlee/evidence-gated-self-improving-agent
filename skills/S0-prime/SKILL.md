# Incident Response Skill

- Find the cause of the reported service degradation and take exactly one disposition that fits that cause.
- Before taking a disposition, observe the evidence it relies on. Do not act on a guess.
- The disposition tools:
  - rollback_deployment rolls back the most recent deployment of the service.
  - scale_service adds capacity to the service.
  - observe_only takes no corrective action and keeps monitoring.
