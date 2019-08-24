from django.db import models

class Random(models.Model):
    string = models.TextField()

    def __str__(self):
        return self.string
