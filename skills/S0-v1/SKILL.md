# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. If a deployment happened around the time of the degradation, roll it back right away. Speed matters more than further checks.
4. Otherwise inspect resource usage and traffic, and scale the service if it is overloaded.
5. Apply the least disruptive action likely to restore service.
