An attempt to train a vision model to identify and mark wall/playable space area on VTT battlemaps and dungeons, thus saving GM's the annoyance of doing it themselves.


Currently a proof of concept model trained just on watabou's one page dungeon generator is working, with tools to take the model mask and output UVTT files with wall positions marked for export to Foundry and other VTT software.

Next steps are to pre-train a model to learn general edge detection on a wide array of battlemaps, then make training masks for 100-200 battlemaps and fine tune the model on those.

Some different art styles might need their own models, i.e. classic OSR monochrome maps.

Apologies for the messy codebase (especially for the watabou only version), this project is in prototype stage and I haven't cleaned out all the old versions and such. Everything to run the final Watabou model is in WatabouTestModel, along with final results.
