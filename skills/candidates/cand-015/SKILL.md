# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. Check dependency health.
4. If there is more than one recent deployment, do not roll the service back; observe and keep monitoring.
5. If a dependency is degraded, do not change the service; observe and keep monitoring.
6. Otherwise inspect resource usage and traffic.