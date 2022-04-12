{	
	"BEGIN": {
		"sent": 
			"[subject.name] wants to go out today.",
		"edge": {
			"location.latent.true": {"LOCNODEC": 1.0, "LOCNODEB": 0.0},
			"location.latent.false": {"LOCNODEC": 0.0, "LOCNODEB": 1.0}
		}
	},

	"LOCNODEB": {
		"sent": 
			"[subject.name] decides to go to the <b>Jungle Brew House</b>.",
		"edge":{
			"location.latent.true": {"NODE11": 0.5, "NODE12": 0.5},
			"location.latent.false": {"NODE11": 0.5, "NODE12": 0.5}
		}
	},
	"LOCNODEC": {
		"sent": 
			"[subject.name] decides to go to the <b>Deep Ocean Cafe</b>.",
		"edge": {
			"location.latent.true": {"NODE11": 0.5, "NODE12": 0.5},
			"location.latent.false": {"NODE11": 0.5, "NODE12": 0.5}
		}
	},

	"NODE11": {
		"sent": 
			"In the <b>[location.name]</b>, [subject.name] waited in line patiently.",
		"edge":{
			"location.latent.true": {"NODE21": 1.0, "NODE22": 0.0},
			"location.latent.false": {"NODE21": 0.0, "NODE22": 1.0}
		}
	},
	"NODE12": {
		"sent": 
			"In the <b>[location.name]</b>, [subject.name] was impatient and decided to cut the line.",
		"edge": {
			"location.latent.true": {"NODE21": 0.0, "NODE22": 1.0},
			"location.latent.false": {"NODE21": 1.0, "NODE22": 0.0}
		}
	},

	"NODE21": {
		"sent": 
			"While ordering tea in the <b>[location.name]</b>, [subject.name] noticed the barista’s new mustache.",
		"edge":{
			"location.latent.true": {"NODE31": 1.0, "NODE32": 0.0},
			"location.latent.false": {"NODE31": 0.0, "NODE32": 1.0}
		}
	},
	"NODE22": {
		"sent": 
			"Before ordering cake in the <b>[location.name]</b>, [subject.name] took a quarter from the tip jar.",
		"edge": {
			"location.latent.true": {"NODE31": 0.0, "NODE32": 1.0},
			"location.latent.false": {"NODE31": 1.0, "NODE32": 0.0}
		}
	},

	"NODE31": {
		"sent": 
			"In the <b>[location.name]</b>, [subject.name] then sat by the window and read a book for hours.",
		"edge": {
			"location.latent.true": {"END": 1.0},
			"location.latent.false": {"END": 1.0}
		}
	},
	"NODE32": {
		"sent": 
			"After ordering, [subject.name] stole a salt shaker and left the <b>[location.name]</b>.",
		"edge": {
			"location.latent.true": {"END": 1.0},
			"location.latent.false": {"END": 1.0}
		}
	},

	
	"END": {
		"sent": 
			"That is all that is remembered.",
		"edge": {}
	}
}