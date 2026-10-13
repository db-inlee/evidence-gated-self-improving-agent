# Incident Response Skill

1. Use service metrics to confirm that the reported service is degraded.
2. Look up recent deployments.
3. Review the health of the dependencies.
4. When there is a recent deployment and every dependency is healthy, roll that deployment back.
5. When a dependency is degraded, leave the service unchanged; observe and keep monitoring.
6. In all other cases, inspect resource usage and traffic.
