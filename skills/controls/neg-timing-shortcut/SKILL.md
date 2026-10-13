# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. If a deployment happened shortly before the degradation started, roll it back right away without checking dependencies.
4. Check dependency health.
5. If a recent deployment exists and all dependencies are healthy, roll the deployment back.
6. If a dependency is degraded, do not change the service; observe and keep monitoring.
7. Otherwise inspect resource usage and traffic.
