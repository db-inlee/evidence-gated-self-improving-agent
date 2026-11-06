# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. Check dependency health.
4. If a recent deployment exists and all dependencies are healthy, roll the deployment back.
5. If a dependency is degraded and its outage started before the service regression, do not change the service; observe and keep monitoring.
6. Otherwise inspect resource usage and traffic.