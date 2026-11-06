# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check dependency health.
3. Check for recent deployments.
4. If a recent deployment exists and all dependencies are healthy, roll the deployment back.
5. If a dependency is degraded, do not change the service; observe and keep monitoring.
6. Otherwise inspect resource usage and traffic.