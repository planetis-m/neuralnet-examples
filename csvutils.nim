import macros, strutils

template genAtom(body: untyped): untyped {.dirty.} =
   let value = nnkBracketExpr.newTree(param, newIntLitNode(pos))
   body
   inc(pos)

proc readType(nodeTy, param: NimNode, pos: var int): NimNode =
   let baseTy = getTypeImpl(nodeTy)
   case baseTy.typeKind
   of ntyRef:
      result = readType(baseTy[0], param, pos)
      if result.kind == nnkObjConstr:
         result[0] = nodeTy
      else:
         error("Only ref objects are supported")
   of ntyObject:
      result = nnkObjConstr.newTree(nodeTy)
      for n in baseTy[2]:
         n.expectKind nnkIdentDefs
         result.add nnkExprColonExpr.newTree(n[0], readType(n[1], param, pos))
   of ntyTuple:
      let isAnonTu = baseTy.kind == nnkTupleConstr
      result = newNimNode(nnkTupleConstr)
      for n in baseTy:
         result.add readType(if isAnonTu: n else: n[1], param, pos)
   of ntyArray:
      result = newNimNode(nnkBracket)
      for i in baseTy[1][1].intVal .. baseTy[1][2].intVal:
         result.add readType(baseTy[2], param, pos)
   of ntyRange:
      result = readType(baseTy[1][1], param, pos)
   of ntyDistinct:
      result = newCall(nodeTy, readType(baseTy[0], param, pos))
   of ntyString:
      genAtom:
         result = value
   of ntyBool:
      genAtom:
         result = newCall(bindSym"parseBool", value)
   of ntyEnum:
      genAtom:
         result = newCall(nnkBracketExpr.newTree(bindSym"parseEnum", nodeTy), value)
   of ntyInt..ntyInt64:
      genAtom:
         result = newCall(baseTy, newCall(bindSym"parseInt", value))
   of ntyFloat..ntyFloat64:
      genAtom:
         result = newCall(baseTy, newCall(bindSym"parseFloat", value))
   of ntyUInt..ntyUInt64:
      genAtom:
         result = newCall(baseTy, newCall(bindSym"parseUInt", value))
   else:
      error("Unsupported type: " & nodeTy.repr)

template checkCompatible(nfields, s): untyped =
   assert nfields == len(s),
      "Type can't be de-serialized from seq, it's fields and seq's length don't match"

macro to*(s: seq[string], T: typedesc): untyped =
   ## De-serializes s to the type specified
   ## case objects, sequences, sets and objects with cycles are unsupported
   let typeSym = getTypeImpl(T)[1]
   var pos = 0 # sequence current index
   let constr = readType(typeSym, s, pos)
   result = nnkStmtListExpr.newTree(
      getAst(checkCompatible(pos, s)),
      constr)
