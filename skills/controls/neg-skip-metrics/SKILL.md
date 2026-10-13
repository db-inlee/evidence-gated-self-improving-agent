# Incident Response Skill

1. Check for recent deployments.
2. Check dependency health.
3. If a recent deployment exists and all dependencies are healthy, roll the deployment back.
4. If a dependency is degraded, do not change the service; observe and keep monitoring.
5. Otherwise inspect resource usage and traffic.
