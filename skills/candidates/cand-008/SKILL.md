# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. Check dependency health.
4. If a recent deployment exists and all dependencies are healthy, roll the deployment back.
5. If a dependency is degraded, compare when the dependency issue began with when the service regression began.
6. Otherwise inspect resource usage and traffic.