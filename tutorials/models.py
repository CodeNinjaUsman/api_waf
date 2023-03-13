from django.db import models


class File(models.Model):
    data = models.TextField()