import pytorch_lightning as L
from pytorch_lightning.demos.boring_classes import BoringModel

model = BoringModel()
trainer = L.Trainer(max_epochs=1000, devices=4, strategy="ddp", logger=None)
trainer.fit(model)