class InterviewState:
    def __init__(self, resume, jd, gap, plan, domain="industry", audit_trace=None):
        self.resume = resume
        self.jd = jd
        self.gap = gap
        self.plan = plan
        self.domain = domain
        self.audit_trace = audit_trace or {}

    def to_dict(self):
        return {
            "resume": self.resume,
            "jd": self.jd,
            "gap": self.gap,
            "plan": self.plan,
            "domain": self.domain,
            "audit_trace": self.audit_trace,
        }
