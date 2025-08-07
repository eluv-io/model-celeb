#!/usr/bin/env node

process.removeAllListeners('warning');
process.on('warning', (warning) => {
  if (warning.code == 'DEP0040') return
  console.warn(warning);
});

const yargs = require('yargs/yargs')
const { hideBin } = require('yargs/helpers')
const { ElvClient } = require("@eluvio/elv-client-js");
const fs = require('fs');
const { v4: UUID, parse: UUIDParse } = require("uuid");
const { parseArgs } = require('util');
const { type } = require('os');

const deepEqual = (a, b) => {
  if (a === b) return true;
  if (!a || !b) return a === b;
  if (typeof a !== 'object' && typeof b !== 'object') return a === b;
  if (a === null || b === null || a === undefined || b === undefined) return false;
  let keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return false;
  return keys.every(key => deepEqual(a[key], b[key]));
};

const argv = yargs(hideBin(process.argv))
      .usage('Usage: $0 <command> [options]')
      .command('migrate-gt <path> <library>', 'migrate ground truth')
      .example('$0 migrate-gt /ml/celeb_pool/sony', 'migrate the sony gt pool')
      .nargs('path', 1)
      .option('finalize', { type: "boolean", default: true})
      .demandOption(['path', 'library'])
      .help('h')
      .alias('h', 'help')
      .parse();

console.log(argv.path)
console.log(argv.library)
console.log(argv.finalize)

const getClient = async () => {
  const privateKey = process.env.PRIVATE_KEY;

  if (!privateKey) {
    throw new Error("PRIVATE_KEY must be set to a valid eluvio private key")
  }

  var client = await ElvClient.FromConfigurationUrl({
    configUrl: "https://main.net955305.contentfabric.io/config",
  });
  const wallet = client.GenerateWallet();
  const signer = wallet.AddAccount({
    privateKey: privateKey,
  });
  client.SetSigner({ signer });
  return client;
};

function getContentType(filename) {
  filename = filename.toLowerCase();
  if (filename.endsWith(".jpg") || filename.endsWith(".jpeg")) return "image/jpeg";
  if (filename.endsWith(".png")) return "image/png";
  throw new Error(`Unknown content type for ${filename}`);
}

async function fabricContentTypeMatching(client, match) {
  const types = await client.ContentTypes()
  console.dir(types, {depth: null})
  let ret = null
  for (const t of Object.values(types)) {
    if ((t.name || "").toLowerCase().includes(match)) {
      if (ret) throw new Error(`There is more than one content type name matching ${match}`)
      ret = t.hash
    }
  }
  return ret;
}

getClient()
  .then(async (client) => {
    try {
      const path = argv.path

      // read id2name to get labels
      const id2namedata = fs.readFileSync(`${path}/id2name.json`, "utf-8")
      const id2name = JSON.parse(id2namedata)

      // everything is created / updated at NOW (this time)
      const timeString = (new Date()).toISOString()

      fileInfo = []
      const entityMeta = {}

      // read all the directories that match sports1234567 format (skip files and other dirs)
      for (const dirname of fs.readdirSync(path)) {        
        if (fs.statSync(`${path}/${dirname}`).isDirectory() && dirname.match(/^[^0-9]+[0-9]{7}/)) {
          console.log(`Processing ${dirname}...`);
          if (id2name[dirname] == null) {
            // this might be a bit too harsh!
            throw new Error(`id2name does not contain ${dirname} -- please update id2name.json`)
          }

          // create data for this whole entity
          // temporarily keep the list of files under "temp"
          const entmeta = {
            temp: {
                files: fs.readdirSync(`${path}/${dirname}`)
                  .filter( f => { return f.includes('.') && ["jpg", "jpeg", "png"].includes(f.toLowerCase().split('.').slice(-1)[0]) } )                
            },
            id: dirname,
            label: id2name[dirname],
            meta: {},
            sample_files: [],
            updated_at: timeString
          }

          // read all the files into buffers in prep for upload
          for (const file of entmeta.temp.files) {
            const data = fs.readFileSync(`${path}/${dirname}/${file}`)
            fileInfo.push({
              path: `/entity_samples/${dirname}/${file}`,
              mime_type: getContentType(file),
              size: data.length,
              data: data,
            });
          }
          entityMeta[dirname] = entmeta
        } else {
          console.log(`Skipping ${dirname}...`);
        }   
      }

      console.log("Creating new object...")
      // create new content object for ground truth
      const gtpool = await client.CreateContentObject({
        "libraryId": argv.library,
        "options": {
          "meta": {
            "public": { 
              "name": "Ground Truth", 
              "description": `Imported legacy Ground Truth pool` 
            },
            //"type": await fabricContentTypeMatching(client, "title")
          },
        }
      });

      console.dir(gtpool, {depth: null})
      console.log(`Setting editable on new object ${gtpool.id}...`)
      await client.SetPermission({
        objectId: gtpool.id,
        permission: "editable",
        writeToken: gtpool.writeToken
      });

      console.log(`Uploading ${fileInfo.length} file(s) to ${gtpool.id}...`)
      await client.UploadFiles({
        libraryId: argv.library,
        objectId: gtpool.id,
        writeToken: gtpool.writeToken,
        fileInfo: fileInfo,
      });


      //const gtpool = {         "id": "iq__8GQqKQYcHkF71rngcg1jdis8sa6",       };   const finalized = {        "id": "iq__8GQqKQYcHkF71rngcg1jdis8sa6",        "hash": "hq__GVGQeNYUJfJXX2c2zDXQkpdJzfnrDy5wrs5tVpoAhcsmE9oCui9pa2AcucJ48YZBQsziyqKVuQ",        "write_token": "tqw__HSQFqWHACekGvvx4MFvZqijgKVWfFDGWSYwGWykYf49Jg36btUEyvkeoT2vCAZwnUVoJbiBGL9jC6BVyyBh",        "type": "",        "qlib_id": "ilib32SQvtQwJ8gaPm3nB2KfMLHzd1Nn",        "object_version": 2}; console.log(JSON.stringify(finalized, null, 2));

      // full metadata under "ground truth" key
      const ground_truth = {
        created_at: timeString,
        updated_at: timeString,
        model_domain: "celebrity_detection", 
        entity_data_schema: {
            properties: {},
            type: "object"
        },
        entities: entityMeta,
      }

      // make sample_files links and ditch the temp key
      for (const [dirname, entmeta] of Object.entries(entityMeta)) {
        entityMeta[dirname].sample_files = []
        for (const file of entmeta.temp.files) {
          entityMeta[dirname].sample_files.push({
            added_at: timeString,
            updated_at: timeString,
            id: client.utils.B58(UUIDParse(UUID())),
            label: file,
            description: `legacy import ${file}`,
            link: {
              "/":  `./files/entity_samples/${dirname}/${file}`
            }
          })                             
        }
        delete entmeta.temp
      }
      


      console.dir(ground_truth, {depth: null})

      if (false) { const editResponse = await client.EditContentObject({
        libraryId: argv.library,
        objectId: gtpool.id,
      });
      }
      
      const editResponse = {
        write_token: gtpool.write_token
      }

      console.log("------------")
      console.log(JSON.stringify(editResponse, null, 2));
      
      console.log(`Setting metadata on new pool ${gtpool.id}...`)

      await client.ReplaceMetadata({libraryId: argv.library, objectId: gtpool.id, writeToken: editResponse.write_token, metadataSubtree: "ground_truth", metadata: ground_truth })
      
      console.log("------------")

      console.log(`Finalizing new pool ${gtpool.id}...`)
      const finalized = await client.FinalizeContentObject({
        libraryId: argv.library,
        objectId: gtpool.id,
        writeToken: editResponse.write_token,
        commitMessage: process.env.COMMIT_MESSAGE || `import legacy gt ${process.env.USER}`,
      });
        
      console.log(JSON.stringify(finalized, null, 2))
    
      return

      const rfinalized = await client.FinalizeContentObject({
        libraryId: argv.library,
        objectId: gtpool.id,
        writeToken: gtpool.writeToken,
        commitMessage: process.env.COMMIT_MESSAGE || `migrate ground truth upload assets ${process.env.USER}`,
      });
      
      return 
      
    } catch (err) {
      console.error(err);
    }
  })
  .catch((err) => {
    console.error(err);
  });
