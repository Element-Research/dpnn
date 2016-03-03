------------------------------------------------------------------------
--[[ Noise Contrast Estimation Criterion ]]--
-- Ref.: A. http://mi.eng.cam.ac.uk/~xc257/papers/ICASSP2015-rnnlm-nce.pdf
--       B. https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf
------------------------------------------------------------------------
local NCECriterion, parent = torch.class("nn.NCECriterion", "nn.Criterion")

function NCECriterion:__init(ncemodule)
   parent.__init(self)
   assert(torch.isTypeOf(ncemodule, 'nn.NCEModule'))
   self.ncemodule = ncemodule     
   self.sizeAverage = true    
end

function NCECriterion:updateOutput(inputTable, target)
   -- P_model(target), P_model(sample), P_noise(target), P_noise(sample)
   local Pmt, Pms, Pnt, Pns = unpack(inputTable)
   local k = self.ncemodule.k
   
   assert(Pmt:dim() == 1)
   assert(Pms:dim() == 2)
   assert(Pnt:dim() == 1)
   assert(Pns:dim() == 2)
   
   -- equation 5 in ref. A
   
   -- eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt) 
   self._Pom = self._Pom or Pmt.new()
   self._Pom:resizeAs(Pmt):copy(Pmt)
   self._Pomdiv = self._Pomdiv or Pmt.new()
   self._Pomdiv:resizeAs(Pmt):copy(Pmt)
   self._Pomdiv:add(k, Pnt)
   self._Pom:cdiv(self._Pomdiv)
   
   -- eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
   self._Pon = self._Pon or Pns.new()
   self._Pon:resizeAs(Pns):copy(Pns):mul(k)
   self._Pondiv = self._Pondiv or Pms.new()
   self._Pondiv:resizeAs(Pms):copy(Pms)
   self._Pondiv:add(k, Pns)
   self._Pon:cdiv(self._Pondiv)
   
   -- equation 6 in ref. A
   
   self._lnPom = self._lnPom or self._Pom.new()
   self._lnPom:log(self._Pom)
   
   self._lnPon = self._lnPon or self._Pon.new()
   self._lnPon:log(self._Pon)
   
   local lnPomsum = self._lnPom:sum()
   local lnPonsum = self._lnPon:sum()
   
   self.output = - (lnPomsum + lnPonsum)
   
   if self.sizeAverage then
      self.output = self.output / Pmt:size(1)
   end
   
   return self.output
end

function NCECriterion:updateGradInput(inputTable, target)

end
